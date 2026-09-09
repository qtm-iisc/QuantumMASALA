"""Validates qtm.symm.rotate_wfn.RotateWfn against genuine Kohn-Sham
wavefunctions from a realistic DFT workflow.

The workflow mirrors how one would check this in practice:

1. Run a proper self-consistent calculation for diamond Si on a uniform
   Monkhorst-Pack grid (as in the 'dft-si/si_scf.py' example), with
   ``symm_rho=True`` so the converged density -- and hence the Kohn-Sham
   potential -- is exactly invariant under the crystal's space group.
2. Fix that converged density (``rho_start=rho``, ``maxiter=1``) and run a
   single non-self-consistent diagonalization at four k-points: k_src;
   k_dest = S @ k_src (mod G), related to k_src by a (nonsymmorphic)
   symmetry operation; Gamma; and -k_src, related to k_src by time reversal
   alone. With ``maxiter=1``, the returned wavefunctions are obtained by
   diagonalizing the Hamiltonian built from the *fixed* input density --
   the density itself is never updated/mixed.
3. Since symmetry-related k-points see the exact same (symmetric)
   Hamiltonian up to the relating operation, RotateWfn.rotate()'s generated
   eigenstates at one must equal the other's eigenstates up to a unitary
   transformation within each degenerate subspace (a single global phase
   in the non-degenerate, 1-fold case). 'RotateWfn' itself has no notion of
   "correctness" to check -- it only ever needs to *generate* wavefunctions
   at the destination k-point, which 'rotate()' does directly, degenerate
   bands and time reversal included. Grouping bands into degenerate
   clusters and checking the resulting representation matrix is unitary is
   purely how *this test* verifies that generation against an
   independently-computed reference; see
   '_degenerate_clusters'/'_assert_rotation_matches' below. This is checked
   for every band, across three scenarios: k_src -> k_dest (a generic,
   nondegenerate symmetry operation), Gamma -> Gamma (self-mapping via a
   nontrivial point-group operation, exercising the 3-fold degenerate
   valence-band-top without spin-orbit), and k_src -> -k_src (time
   reversal alone, exercising 'rotate''s antiunitary branch).
"""
import os

import numpy as np
import spglib
import spgrep

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED
from qtm.constants import RYDBERG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, scf
from qtm.gspace import GSpace
from qtm.io_utils.dft_printers import print_scf_status, print_eigenvalues
from qtm.kpts import gen_monkhorst_pack_grid, KList
from qtm.lattice import RealLattice, ReciLattice
from qtm.mpi import QTMComm
from qtm.pseudo import UPFv2Data
from qtm.symm.rotate_wfn import RotateWfn

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

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

# scf(symm_rho=True) symmetrizes the density with a 'SymmFieldMod', which
# drops symmetry ops whose fractional translation isn't commensurate with
# grho's FFT grid ('crystal.symm.filter_frac_trans', called internally).
# Diamond Si's nonsymmorphic operations carry tau=(1/4,1/4,1/4)-type
# translations, so this requires grid dimensions divisible by 4 -- with an
# incommensurate grid, only a symmetry SUBGROUP survives (here: order 12,
# whose largest irrep is 2-dimensional instead of the full point group's
# 3-dimensional one), and bands that should be exactly degenerate come out
# of SCF with a small but very real (not just numerical-noise) spurious
# splitting -- e.g. ~1e-6 Ha for Gamma's valence-band-top triplet at
# ecut_wfn=8 Ry (grid (15,15,15), only 12/48 ops survive), three orders of
# magnitude worse than at ecut_wfn=16 Ry (grid (20,20,20), all 48 survive).
# Picking a grid-commensurate ecut_wfn keeps the degeneracies below exactly
# what they should be, so this test can use a much tighter tolerance.
ecut_wfn = 16 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho
assert all(n % 4 == 0 for n in grho.grid_shape), grho.grid_shape

crystal.symm.filter_frac_trans(grho.grid_shape)
assert crystal.symm.numsymm == 48, crystal.symm.numsymm

# ----- Step 1: converge the charge density on a uniform MP grid -----------
mpgrid_shape = (5, 5, 5)
mpgrid_shift = (True, True, True)
kpts_scf = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

numbnd_scf = crystal.numel // 2  # occupied bands only, for the density

scf_out = scf(
    dftcomm,
    crystal,
    kpts_scf,
    grho,
    gwfn,
    numbnd_scf,
    is_spin=False,
    is_noncolin=False,
    symm_rho=True,
    rho_start=None,
    occ_typ="fixed",
    conv_thr=1e-12 * RYDBERG,
    diago_thr_init=1e-2 * RYDBERG,
    iter_printer=print_scf_status,
)
scf_converged, rho_scf, _l_wfn_kgrp_scf, _en_scf = scf_out

# ----- Step 2: fix that density and diagonalize at two symmetry-related ---
# ----- k-points in a single (non-self-consistent) iteration ---------------
rho = rho_scf.copy()
k_src = (0.13, 0.38, -0.08)
# Pick the first symmetry after identity
isymm = 1

R_recip = crystal.symm.recilat_rot[isymm]
k_rot = R_recip @ np.array(k_src)
k_dest = tuple(k_rot - np.rint(k_rot))

# The Gamma point is fixed by *every* symmetry operation (S @ 0 = 0 for any
# S), so 'isymm' also relates it to itself -- since Gamma has a nontrivial
# little group, silicon's valence-band-top there is (without spin-orbit)
# 3-fold degenerate, giving a genuine degenerate subspace to rotate.
k_gamma = (0.0, 0.0, 0.0)

# k_src -> -k_src is a genuine time-reversal pair. Diamond Si is
# centrosymmetric, so an *unconstrained* symmetry search would find a
# perfectly good non-time-reversal path too (inversion alone also maps
# k -> -k); 'time_reversal=True' is passed explicitly below to force
# exercising the antiunitary (complex-conjugating) code path specifically,
# using real SCF wavefunctions rather than the synthetic ones in
# 'test_rotate_wfn.py::test_time_reversal'.
k_src_tr = tuple(-np.array(k_src))

kpts_nscf = KList(
    crystal.recilat,
    np.array([k_src, k_dest, k_gamma, k_src_tr]).T,
    np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]),
)

numbnd = crystal.numel // 2 + 5  # a few empty bands, plus one extra "buffer" band
NUMBND_CHECK = numbnd - 1  # the topmost requested band converges least reliably

nscf_out = scf(
    dftcomm,
    crystal,
    kpts_nscf,
    grho,
    gwfn,
    numbnd,
    is_spin=False,
    is_noncolin=False,
    symm_rho=True,
    rho_start=rho,  # fixed, converged density from step 1
    occ_typ="fixed",
    maxiter=1,  # a single diagonalization; the density is never updated
    diago_thr_init=1e-12 * RYDBERG,  # tight, since there is no 2nd iteration
    iter_printer=print_scf_status,
)
_nscf_converged, _rho_nscf, l_wfn_kgrp, _en_nscf = nscf_out

# With a single k-group (serial run), 'l_wfn_kgrp' is ordered exactly as 'kpts_nscf'
wfn_src = l_wfn_kgrp[0][0]
wfn_dest = l_wfn_kgrp[1][0]
wfn_gamma = l_wfn_kgrp[2][0]
wfn_src_tr = l_wfn_kgrp[3][0]
assert wfn_src.k_cryst == k_src
assert np.allclose(wfn_dest.k_cryst, k_dest)
assert wfn_gamma.k_cryst == k_gamma
assert np.allclose(wfn_src_tr.k_cryst, k_src_tr)

rot = RotateWfn(crystal, wfn_src.gkspc, wfn_dest.gkspc, isymm=isymm)
rot_gamma = RotateWfn(crystal, wfn_gamma.gkspc, wfn_gamma.gkspc, isymm=isymm)
rot_tr = RotateWfn(crystal, wfn_src.gkspc, wfn_src_tr.gkspc, time_reversal=True)

TOL_DEG = 1e-6 * RYDBERG  # eigenvalue gap below which bands are (near-)degenerate
ATOL = 1e-6  # tolerance on the representation matrix's deviation from unitary
# With the grid-commensurate 'ecut_wfn' above, a symmetry-protected
# degenerate manifold's residual splitting from a single (from-random,
# maxiter=1) Davidson diagonalization is ~1e-8 Ha (vs. ~1e-6 Ha with the
# incommensurate grid this test used previously) -- TOL_DEG only needs to
# comfortably exceed that, while staying far below genuine inter-level gaps
# (~0.1 Ha), so it can be much tighter than before.


# The topmost few bands of a fixed-size Davidson diagonalization converge
# less reliably than the rest (worst for the very last one), especially
# from a random initial guess in a single (maxiter=1) iteration; comparisons
# below are restricted to the lower 'NUMBND_CHECK' bands, treating the
# highest requested band(s) as unchecked convergence "buffer" -- standard
# practice for validating band-structure-type calculations.
def _checked(wfn):
    return wfn.evc_gk[:NUMBND_CHECK], wfn.evl[:NUMBND_CHECK]


# ---------------------------------------------------------------------------
# spgrep-based degeneracy prediction: instead of *trusting* an eigenvalue-gap
# tolerance to have found the right clusters (which we saw can go wrong --
# an incommensurate FFT grid silently split a true 3-fold degeneracy into
# 2+1, both individually "plausible" sizes), get the set of degeneracy
# dimensions the crystal's space group actually allows at a k-point from
# group theory alone, and use it to catch cluster sizes that are outright
# impossible, plus confirm the maximal allowed degeneracy is actually
# realized somewhere. This does not replace eigenvalue gaps as the
# mechanism that groups bands (that still requires the numerical data --
# spgrep has no notion of "which particular bands"), but it removes the
# need to hand-verify (as we did earlier, in a one-off script) what
# clusters *should* be possible for this material.
def _spglib_cell(crystal: Crystal):
    """(lattice, positions, numbers) for spglib/spgrep, in spglib's cell
    convention (row-wise lattice vectors). Built fresh from
    'crystal.reallat'/'crystal.l_atoms' every time -- 'crystal' (including
    whatever 'crystal.symm.filter_frac_trans' has done to 'crystal.symm')
    is only ever read here, never modified.
    """
    lattice = np.array(crystal.reallat.latvec.T)
    positions, numbers = [], []
    for ityp, sp in enumerate(crystal.l_atoms):
        positions.append(np.array(sp.r_cryst.T))
        numbers += [ityp] * sp.r_cryst.shape[1]
    return lattice, np.concatenate(positions, axis=0), np.array(numbers)


def _spgrep_allowed_dims(crystal: Crystal, k_cryst) -> list[int]:
    """Every band-degeneracy dimension the crystal's space group allows at
    k-point 'k_cryst', from group theory alone (no DFT).

    'k_cryst' is given, and this returns, entirely in terms of
    QuantumMASALA's own 'crystal.recilat' crystal-coordinate convention.
    Internally, this runs on a private copy of the structure
    ('_spglib_cell') reduced to spglib's own choice of primitive cell via
    'spglib.find_primitive' -- which can be a different (though
    equivalent, related by a unimodular basis change) primitive cell than
    QuantumMASALA's own -- so 'k_cryst' is transformed into that basis
    (via Cartesian coordinates, reusing QuantumMASALA's own
    'cryst2cart'/'cart2cryst' for both lattices to guarantee a consistent
    round trip) before calling spgrep. 'crystal' itself is never modified.
    """
    symprec = crystal.symm.symprec
    cell = _spglib_cell(crystal)
    prim_lattice, prim_positions, prim_numbers = spglib.find_primitive(
        cell, symprec=symprec
    )

    # A throwaway QuantumMASALA lattice pair for spglib's primitive cell,
    # purely to reuse 'cryst2cart'/'cart2cryst' for a correctly-normalized
    # change of basis -- 'crystal.reallat' itself is untouched.
    prim_reallat = RealLattice(
        alat=1.0, latvec=np.ascontiguousarray(prim_lattice.T, dtype="f8")
    )
    prim_recilat = ReciLattice.from_reallat(prim_reallat)

    k_cart = crystal.recilat.cryst2cart(np.asarray(k_cryst, dtype="f8"))
    k_prim = prim_recilat.cart2cryst(k_cart)

    sym = spglib.get_symmetry(
        (prim_lattice, prim_positions, prim_numbers), symprec=symprec
    )
    irreps, _ = spgrep.get_spacegroup_irreps_from_primitive_symmetry(
        sym["rotations"], sym["translations"], k_prim
    )
    return sorted({irrep.shape[-1] for irrep in irreps})


def _degenerate_clusters(evl, tol_deg, allowed_dims=None):
    """Groups band indices into runs of mutually (near-)degenerate bands.

    Within a d-fold degenerate eigenspace, a symmetry operation does not
    map individual eigenvectors to individual eigenvectors -- it acts on
    the whole eigenspace as a d-dimensional (generically irreducible)
    representation, since diagonalizing a Hamiltonian only determines a
    degenerate eigenspace, not any particular orthonormal basis of it.

    If 'allowed_dims' is given (see '_spgrep_allowed_dims'), every
    resulting cluster's size is checked against it: a size spgrep says is
    group-theoretically impossible at this k-point means 'tol_deg' is
    wrong for this data, or (as happened with an FFT grid incommensurate
    with the space group -- see 'ecut_wfn' above) a symmetry-protected
    degeneracy has been spuriously split.
    """
    clusters = [[0]]
    for n in range(1, len(evl)):
        if abs(evl[n] - evl[n - 1]) < tol_deg:
            clusters[-1].append(n)
        else:
            clusters.append([n])

    if allowed_dims is not None:
        for cluster in clusters:
            assert len(cluster) in allowed_dims, (
                f"cluster {cluster} has size {len(cluster)}, not among the "
                f"degeneracy dimensions {allowed_dims} spgrep says this "
                f"k-point's little group allows"
            )
    return clusters


def _assert_rotation_matches(rot, wfn_a, wfn_b, tol_deg=TOL_DEG, atol=ATOL):
    """Checks that ``rot.rotate(wfn_a.evc_gk)`` reproduces ``wfn_b.evc_gk``,
    cluster-by-cluster of degenerate bands.

    For a d-fold degenerate cluster, individual rotated eigenvectors need
    not match individual dest eigenvectors band-by-band -- only up to an
    overall phase in the non-degenerate (d=1) case. What must hold in
    general is that the d x d matrix of overlaps between the rotated src
    eigenvectors and the dest eigenvectors,
    ``D[i, j] = <dest_i | rotate(src)_j>`` (the representation matrix of
    the symmetry operation on that eigenspace), is unitary -- i.e.
    rotation maps the src eigenspace exactly onto the dest eigenspace,
    regardless of which orthonormal basis either was diagonalized to.
    """
    evc_a, evl_a = _checked(wfn_a)
    evc_b, evl_b = _checked(wfn_b)
    assert np.allclose(evl_a, evl_b, atol=tol_deg)

    allowed_dims = _spgrep_allowed_dims(crystal, wfn_a.k_cryst)
    data_rot = rot.rotate(evc_a).data
    data_b = evc_b.data
    for cluster in _degenerate_clusters(evl_a, tol_deg, allowed_dims):
        idx = np.array(cluster)
        d_mat = data_b[idx].conj() @ data_rot[idx].T  # <b_i | rotate(a)_j>
        assert np.allclose(
            d_mat @ d_mat.conj().T, np.eye(len(cluster)), atol=atol
        ), f"representation matrix not unitary for bands {cluster}"


def test_scf_converged():
    assert scf_converged


def test_rotated_scf_wavefunctions_match_up_to_unitary_rotation():
    _assert_rotation_matches(rot, wfn_src, wfn_dest)


def test_rotated_scf_wavefunctions_at_gamma_match_up_to_unitary_rotation():
    # Self-mapping via a nontrivial point-group operation: exercises the
    # degenerate-subspace handling on silicon's 3-fold degenerate valence
    # band top at Gamma (bands 1, 2, 3: the lowest band, 0, is the
    # non-degenerate s-like state below it). The expected maximal
    # degeneracy (3) is derived from spgrep, not hardcoded, so this would
    # also catch e.g. the incommensurate-grid regression from earlier.
    _, evl_gamma = _checked(wfn_gamma)
    allowed_dims = _spgrep_allowed_dims(crystal, k_gamma)
    clusters = _degenerate_clusters(evl_gamma, TOL_DEG, allowed_dims)
    max_found = max(len(c) for c in clusters)
    assert max_found == max(allowed_dims), (
        f"expected the maximal spgrep-allowed degeneracy {max(allowed_dims)} "
        f"(from {allowed_dims}) to be realized at Gamma, but the largest "
        f"cluster found has only {max_found} bands: {clusters} from "
        f"eigenvalues {evl_gamma}"
    )
    _assert_rotation_matches(rot_gamma, wfn_gamma, wfn_gamma)


def test_rotated_scf_wavefunctions_match_time_reversal():
    # k_src -> -k_src via time reversal alone (isymm forced to identity's
    # combination with time reversal, not the inversion-based non-TR path
    # that would also relate them in this centrosymmetric crystal),
    # exercising the antiunitary (complex-conjugating) branch of 'rotate'
    # against real SCF wavefunctions.
    assert rot_tr.time_reversal is True
    _assert_rotation_matches(rot_tr, wfn_src, wfn_src_tr)
