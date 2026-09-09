"""Simple, DFT-free unit tests for `qtm.kpts` (the actively-used k-point
list/Monkhorst-Pack grid machinery) and a regression test for a bug fixed in
`qtm.klist.KList.mpgrid` (a separate, legacy `KList` implementation still
used by the GW module, but whose `.mpgrid` classmethod itself has no
callers anywhere in the codebase).

Uses a synthetic simple-cubic 1-atom crystal (as in
'../crystal_tests/test_crystal.py') -- no pseudopotential, no SCF.
"""
import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.crystal import BasisAtoms, Crystal
from qtm.klist import KList as KListLegacy
from qtm.kpts import KList, gen_monkhorst_pack_grid
from qtm.lattice import RealLattice

reallat = RealLattice.from_alat(alat=5.0, a1=[1.0, 0, 0], a2=[0, 1.0, 0], a3=[0, 0, 1.0])
atoms = BasisAtoms("X", 4, None, reallat, np.array([[0.0], [0.0], [0.0]]))
crystal = Crystal(reallat, [atoms])


def _manual_mp_grid(grid_shape, shifts, is_time_reversal):
    if is_time_reversal:
        ki = [
            (np.arange(ni // 2 + 1) + 0.5 * si) / ni
            for ni, si in zip(grid_shape, shifts)
        ]
    else:
        ki = [
            (np.arange(-ni // 2 + 1, ni // 2 + 1) + 0.5 * si) / ni
            for ni, si in zip(grid_shape, shifts)
        ]
    mesh = np.meshgrid(*ki, indexing="ij")
    return np.transpose([np.ravel(a) for a in mesh])


# ----- qtm.kpts.gen_monkhorst_pack_grid -------------------------------------
def test_no_symm_matches_closed_form_formula_and_uniform_weights():
    grid_shape, shifts = (4, 3, 2), (False, False, False)
    kl = gen_monkhorst_pack_grid(
        crystal, grid_shape, shifts, use_symm=False, is_time_reversal=True
    )

    manual = _manual_mp_grid(grid_shape, shifts, is_time_reversal=True)
    assert set(map(tuple, np.round(manual, 10))) == set(
        map(tuple, np.round(kl.k_cryst.T, 10))
    )
    assert np.allclose(kl.k_weights, 1.0 / kl.numkpts)
    assert np.isclose(np.sum(kl.k_weights), 1.0)


def test_time_reversal_restricts_grid_to_half_the_full_mesh():
    grid_shape, shifts = (4, 4, 4), (False, False, False)
    kl_tr = gen_monkhorst_pack_grid(
        crystal, grid_shape, shifts, use_symm=False, is_time_reversal=True
    )
    kl_notr = gen_monkhorst_pack_grid(
        crystal, grid_shape, shifts, use_symm=False, is_time_reversal=False
    )
    assert kl_tr.numkpts == 3 * 3 * 3  # (ni//2 + 1) per dimension
    assert kl_notr.numkpts == 4 * 4 * 4  # the full uniform mesh
    assert np.isclose(np.sum(kl_tr.k_weights), 1.0)
    assert np.isclose(np.sum(kl_notr.k_weights), 1.0)


def test_symm_reduced_gamma_only_grid_is_a_single_point_of_weight_one():
    kl = gen_monkhorst_pack_grid(
        crystal, (1, 1, 1), (False, False, False), use_symm=True
    )
    assert kl.numkpts == 1
    assert np.allclose(kl.k_cryst, 0.0)
    assert np.isclose(kl.k_weights[0], 1.0)


def test_symm_reduction_never_increases_kpoint_count():
    grid_shape, shifts = (4, 4, 4), (False, False, False)
    kl_symm = gen_monkhorst_pack_grid(crystal, grid_shape, shifts, use_symm=True)
    kl_full = gen_monkhorst_pack_grid(crystal, grid_shape, shifts, use_symm=False)
    assert kl_symm.numkpts <= kl_full.numkpts
    assert np.isclose(np.sum(kl_symm.k_weights), 1.0)
    assert np.all(kl_symm.k_weights > 0)


# ----- qtm.kpts.KList ----------------------------------------------------------
def test_klist_gamma_and_cart_tpiba_consistency():
    kl = KList.gamma(crystal.recilat)
    assert kl.numkpts == 1 and len(kl) == 1
    assert np.allclose(kl.k_cart, 0.0)
    assert np.allclose(kl.k_tpiba, 0.0)


def test_klist_getitem_int_vs_slice():
    k_coords = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]).T
    weights = np.array([0.5, 0.25, 0.25])
    kl = KList(crystal.recilat, k_coords, weights)

    k, w = kl[1]
    assert np.allclose(k, (0.0, 0.0, 0.0)) and np.isclose(w, 0.25)

    sub = kl[0:2]
    assert isinstance(sub, KList) and sub.numkpts == 2


def test_klist_scatter_partitions_all_kpoints_exactly_once():
    numk = 7
    k_coords = np.stack([np.arange(numk, dtype="f8") / numk] * 3)
    weights = np.ones(numk) / numk
    kl = KList(crystal.recilat, k_coords, weights)

    n_kgrp = 3
    seen = []
    for i_kgrp in range(n_kgrp):
        part = kl.scatter(n_kgrp, i_kgrp)
        seen.append(part.k_cryst)
    total = sum(p.shape[1] for p in seen)
    assert total == numk
    all_k = np.concatenate(seen, axis=1)
    assert set(map(tuple, np.round(all_k.T, 10))) == set(
        map(tuple, np.round(kl.k_cryst.T, 10))
    )


# ----- regression: qtm.klist.KList.mpgrid (legacy class) ---------------------
def test_legacy_klist_mpgrid_matches_gen_monkhorst_pack_grid_after_fix():
    # 'KList.mpgrid's non-symmetry branch used to compute
    # 'np.arange(ni // 2 + 1) + 0.5 * si / ni' (missing parens around the
    # 'arange' term, so it was never divided by 'ni' into fractional crystal
    # coordinates) -- compare directly against the correct, actively-used
    # 'qtm.kpts.gen_monkhorst_pack_grid' implementation for both
    # 'is_time_reversal' branches to pin the fix down.
    grid_shape, shifts = (3, 2, 2), (False, False, False)
    for is_time_reversal in (True, False):
        legacy = KListLegacy.mpgrid(
            crystal,
            grid_shape,
            shifts,
            use_symm=False,
            is_time_reversal=is_time_reversal,
        )
        correct = gen_monkhorst_pack_grid(
            crystal,
            grid_shape,
            shifts,
            use_symm=False,
            is_time_reversal=is_time_reversal,
        )
        set_legacy = set(map(tuple, np.round(legacy.cryst, 10)))
        set_correct = set(map(tuple, np.round(correct.k_cryst.T, 10)))
        assert set_legacy == set_correct
