"""Validates qtm.symm.symmetrize_field.SymmFieldMod against two ground-truth
physical invariants of symmetrization: idempotency, and invariance of the
symmetrized field under the crystal's space-group operations, checked
directly in real space (in the same spirit as
'test_rotate_wfn.py::test_real_space_transformation_law').

Uses the same diamond-Si crystal and 'ecut' as 'test_rotate_wfn.py' (a
nonsymmorphic space group, with an FFT grid divisible by 4 so that all 48
point-group operations survive 'filter_frac_trans' -- see
'test_rotate_wfn_scf.py' for why an incommensurate grid would matter here).
Unlike a Bloch wavefunction, a 'FieldG' has no k-point, so its real-space
invariance law has no extra Bloch phase: f(r) = f(S^-1(r - tau)) exactly.
"""
import os

import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.containers import get_FieldG
from qtm.crystal import BasisAtoms, Crystal
from qtm.gspace import GSpace
from qtm.lattice import RealLattice
from qtm.pseudo import UPFv2Data
from qtm.symm.symmetrize_field import SymmFieldMod

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

ecut = 40.0
gwfn = GSpace(crystal.recilat, ecut)
grid_shape = gwfn.grid_shape
grid_dims = np.array(grid_shape)

crystal.symm.filter_frac_trans(grid_shape)
assert crystal.symm.numsymm == 48, crystal.symm.numsymm

symmod = SymmFieldMod(crystal, gwfn)
FieldG = get_FieldG(gwfn)

rng = np.random.default_rng(0)


def _random_field(shape):
    data = rng.standard_normal((*shape, gwfn.size_g)) + 1j * rng.standard_normal(
        (*shape, gwfn.size_g)
    )
    return FieldG.from_array(data)


def test_symmetrize_does_not_mutate_input():
    f = _random_field((2,))
    orig = f.copy()
    symmod.symmetrize(f)
    assert np.array_equal(f.data, orig.data)


def test_symmetrize_is_idempotent():
    f = _random_field((2,))
    once = symmod.symmetrize(f)
    twice = symmod.symmetrize(once)
    assert np.allclose(once.data, twice.data, atol=1e-10)


def test_symmetrize_is_invariant_under_crystal_symmetries():
    f = symmod.symmetrize(_random_field((1,)))
    u_r = f.to_r().data.reshape(1, *grid_shape)

    rng_pts = np.random.default_rng(1)
    # A handful of symmetry operations (including a couple of the
    # crystal's genuinely nonsymmorphic ones) x random real-space points.
    nonsymmorphic = [
        i
        for i in range(crystal.symm.numsymm)
        if not np.allclose(crystal.symm.reallat_trans[i], 0.0)
    ]
    isymm_list = [0] + nonsymmorphic[:2] + [crystal.symm.numsymm - 1]

    for isymm in isymm_list:
        R_real = crystal.symm.reallat_rot[isymm]
        tau = crystal.symm.reallat_trans[isymm]
        Rinv_real = np.rint(np.linalg.inv(R_real)).astype("i8")
        assert np.allclose(Rinv_real @ R_real, np.eye(3))

        tau_grid = tau * grid_dims
        assert np.allclose(tau_grid, np.rint(tau_grid), atol=1e-6)
        tau_grid = np.rint(tau_grid).astype("i8")

        for _ in range(50):
            idx = rng_pts.integers(0, grid_dims)
            val_dest = u_r[:, idx[0], idx[1], idx[2]]

            idx_src = (Rinv_real @ (idx - tau_grid)) % grid_dims
            val_src = u_r[:, idx_src[0], idx_src[1], idx_src[2]]

            assert np.allclose(val_dest, val_src, atol=1e-8), (
                f"symmetrized field not invariant under isymm={isymm} "
                f"at grid point {idx}"
            )
