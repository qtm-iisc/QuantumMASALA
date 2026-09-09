"""Validates qtm.symm.rotate_wfn.RotateWfn against a direct real-space
evaluation of the Bloch-wavefunction symmetry transformation law, using a
diamond-Si crystal (nonsymmorphic space group, with genuine tau=(0,1/2,1/2)
type fractional translations).

For a symmetry operation g = {S | tau} of the crystal (S = recilat_rot,
acting on reciprocal-space crystal coordinates; tau = reallat_trans, a
real-space fractional translation), the exact statement being checked is

    psi_dest(r) = psi_src(g^-1 r),   g^-1: r -> R_real^-1 (r - tau)

where R_real = reallat_rot is the corresponding real-space rotation. This
is verified directly on the real-space FFT grid, including the Bloch phase
picked up when g^-1 maps a grid point outside of the [0, n) unit cell.
"""
import os

import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import TPI
from qtm.containers import get_WavefunG
from qtm.crystal import BasisAtoms, Crystal
from qtm.gspace import GkSpace, GSpace
from qtm.lattice import RealLattice
from qtm.pseudo import UPFv2Data
from qtm.symm.rotate_wfn import RotateWfn

# Lattice and pseudopotential (reusing the Si UPF file from 'system_tests')
reallat = RealLattice.from_alat(
    alat=10.2, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]  # Bohr
)
si_oncv = UPFv2Data.from_file(
    os.path.join(
        os.path.dirname(__file__), "..", "system_tests", "Si_ONCV_PBE-1.2.upf"
    )
)
si_atoms = BasisAtoms(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]).T,
)
crystal = Crystal(reallat, [si_atoms])

ecut = 40.0  # Ry
gwfn = GSpace(crystal.recilat, ecut)
grid_shape = gwfn.grid_shape
grid_dims = np.array(grid_shape)

# Only symmetry ops whose fractional translation is commensurate with the
# FFT grid can be checked exactly on real-space grid points.
crystal.symm.filter_frac_trans(grid_shape)

# A generic (low-symmetry) k-point, and a nonsymmorphic operation (tau != 0)
# relating it to a symmetry-equivalent k-point.
k_src = (0.13, 0.27, -0.08)

isymm_expected = None
for isymm in range(crystal.symm.numsymm):
    if not np.allclose(crystal.symm.reallat_trans[isymm], 0.0):
        isymm_expected = isymm
        break
assert isymm_expected is not None, "no nonsymmorphic operation with tau != 0 found"

R_recip = crystal.symm.recilat_rot[isymm_expected]
R_real = crystal.symm.reallat_rot[isymm_expected]
tau = crystal.symm.reallat_trans[isymm_expected]

k_rot = R_recip @ np.array(k_src)
k_dest = tuple(k_rot - np.rint(k_rot))  # fold to a canonical label

gkspc_src = GkSpace(gwfn, k_src)
gkspc_dest = GkSpace(gwfn, k_dest)

rot = RotateWfn(crystal, gkspc_src, gkspc_dest)

nbnd = 4
np.random.seed(0)
WavefunG_src = get_WavefunG(gkspc_src, 1)
wfn_src = WavefunG_src.empty(nbnd)
wfn_src.data[:] = np.random.randn(nbnd, gkspc_src.size_g) + 1j * np.random.randn(
    nbnd, gkspc_src.size_g
)
wfn_src.normalize()

wfn_dest = rot.rotate(wfn_src)

u_src_r = wfn_src.to_r().data.reshape(nbnd, *grid_shape)
u_dest_r = wfn_dest.to_r().data.reshape(nbnd, *grid_shape)


def test_symmetry_search_finds_expected_operation():
    assert rot.isymm == isymm_expected
    assert rot.time_reversal is False


def test_norm_preserved():
    # Rotating a wavefunction between symmetry-related k-points is unitary
    assert np.allclose(wfn_src.norm2(), wfn_dest.norm2(), atol=1e-10)


def test_real_space_transformation_law():
    # psi_dest(r) = psi_src(g^-1 r), g^-1: r -> R_real^-1 (r - tau)
    Rinv_real = np.rint(np.linalg.inv(R_real)).astype("i8")
    assert np.allclose(Rinv_real @ R_real, np.eye(3))

    tau_grid = tau * grid_dims
    assert np.allclose(
        tau_grid, np.rint(tau_grid), atol=1e-6
    ), "tau not grid-commensurate"
    tau_grid = np.rint(tau_grid).astype("i8")

    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(200):
        idx = rng.integers(0, grid_dims)  # (i1, i2, i3) on the dest grid
        phase_dest = np.exp(1j * TPI * np.sum(np.array(k_dest) * idx / grid_dims))
        psi_dest = phase_dest * u_dest_r[:, idx[0], idx[1], idx[2]]

        # When g^-1 maps 'idx' outside the [0, n) cell, the wrapped-back
        # point picks up an extra Bloch phase e^{i*2pi*k_src.m} for the
        # integer winding vector 'm' (psi_k(r + m) = e^{i*2pi*k.m} psi_k(r))
        idx_src_raw = Rinv_real @ (idx - tau_grid)
        m = np.floor(idx_src_raw / grid_dims).astype("i8")
        idx_src = idx_src_raw - m * grid_dims
        phase_src = np.exp(1j * TPI * np.sum(np.array(k_src) * idx_src / grid_dims))
        phase_wind = np.exp(1j * TPI * np.sum(np.array(k_src) * m))
        psi_src = (
            phase_wind * phase_src * u_src_r[:, idx_src[0], idx_src[1], idx_src[2]]
        )

        max_err = max(max_err, np.max(np.abs(psi_dest - psi_src)))

    assert max_err < 1e-8, "real-space transformation law violated"


def test_round_trip():
    # Rotating back recovers the original wavefunction up to a global phase.
    # (The symmetry op found for the return trip need not be the exact group
    # inverse of 'isymm_expected' as a Seitz operator -- spglib reports
    # translations mod a lattice vector, and {R|tau} vs {R|tau+t} act
    # identically on lattice-periodic fields but differ by the Bloch phase
    # e^{-i*2pi*k.t} on a wavefunction, so the round trip is only guaranteed
    # up to such an unobservable global phase.)
    rot_back = RotateWfn(crystal, gkspc_dest, gkspc_src)
    wfn_back = rot_back.rotate(wfn_dest)
    ratio = wfn_back.data[0] / wfn_src.data[0]
    assert np.allclose(ratio, ratio[0], atol=1e-10)
    assert np.isclose(abs(ratio[0]), 1.0, atol=1e-10)


def test_time_reversal():
    # psi_dest(r) = conj(psi_src(r)) under k -> -k combined with time reversal.
    # Diamond Si is centrosymmetric, so an unconstrained search finds a
    # genuine (non-time-reversal) inversion-type operation mapping k_src to
    # -k_src first; force 'time_reversal=True' to specifically exercise the
    # time-reversal code path.
    k_dest_tr = tuple(-np.array(k_src))
    gkspc_tr = GkSpace(gwfn, k_dest_tr)
    rot_tr = RotateWfn(crystal, gkspc_src, gkspc_tr, time_reversal=True)
    assert rot_tr.time_reversal is True

    wfn_tr = rot_tr.rotate(wfn_src)
    u_tr_r = wfn_tr.to_r().data.reshape(nbnd, *grid_shape)

    rng = np.random.default_rng(2)
    max_err = 0.0
    for _ in range(200):
        idx = rng.integers(0, grid_dims)
        phase_tr = np.exp(1j * TPI * np.sum(np.array(k_dest_tr) * idx / grid_dims))
        psi_tr = phase_tr * u_tr_r[:, idx[0], idx[1], idx[2]]

        phase_src = np.exp(1j * TPI * np.sum(np.array(k_src) * idx / grid_dims))
        psi_src = phase_src * u_src_r[:, idx[0], idx[1], idx[2]]

        max_err = max(max_err, np.max(np.abs(psi_tr - np.conj(psi_src))))

    assert max_err < 1e-8
