"""Simple, DFT-free unit tests for `qtm.gw.core`: `QPoints`'s Gamma-point
auto-detection, `sort_cryst_like_BGW`'s 4-key lexsort tie-breaking
convention, and `reorder_2d_matrix_sorted_gvecs`'s simultaneous row/column
permutation.

Uses only a bare `ReciLattice` and plain numpy arrays -- no GSpace,
Crystal, or pseudopotential needed at all.
"""
import numpy as np

from qtm.gw.core import QPoints, reorder_2d_matrix_sorted_gvecs, sort_cryst_like_BGW
from qtm.lattice import ReciLattice

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))


def test_qpoints_autodetects_gamma_by_norm_when_is_q0_not_given():
    cryst = np.array([[0.3, 0.1, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    qpts = QPoints(recilat, None, cryst)
    assert qpts.index_q0 == 1
    assert np.allclose(qpts.q0vec, [0.0, 0.0, 0.0])


def test_qpoints_uses_explicit_is_q0_flags():
    cryst = np.array([[0.3, 0.1, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    is_q0 = [False, False, True]  # deliberately NOT the norm-closest point
    qpts = QPoints(recilat, is_q0, cryst)
    assert qpts.index_q0 == 2


def test_sort_cryst_like_bgw_matches_independent_tuple_sort():
    rng = np.random.default_rng(0)
    n = 15
    cryst = rng.integers(-2, 3, size=(3, n))
    key = np.sum(cryst**2, axis=0).astype(float)

    idx = sort_cryst_like_BGW(cryst, key)

    # Independent ground truth: priority is key > cryst[0] > cryst[1] > cryst[2]
    # (np.lexsort's LAST key argument is the primary sort key).
    expected = sorted(
        range(n), key=lambda i: (round(key[i], 10), cryst[0, i], cryst[1, i], cryst[2, i])
    )
    assert list(idx) == expected


def test_sort_cryst_like_bgw_breaks_ties_by_cryst_components_in_order():
    # All four vectors have the same norm^2 (a genuine tie on the primary
    # key), so the order must be decided entirely by cryst[0], then
    # cryst[1], then cryst[2].
    cryst = np.array(
        [
            [1, -1, 0, 0],
            [0, 0, 1, -1],
            [0, 0, 0, 0],
        ]
    )
    key = np.ones(4)
    idx = sort_cryst_like_BGW(cryst, key)
    assert list(cryst[0, idx]) == [-1, 0, 0, 1]
    assert list(cryst[1, idx]) == [0, -1, 1, 0]


def test_reorder_2d_matrix_matches_np_ix_ground_truth():
    rng = np.random.default_rng(1)
    n = 6
    mat = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    indices = rng.permutation(n)

    out = reorder_2d_matrix_sorted_gvecs(mat, indices)
    expected = mat[np.ix_(indices, indices)]
    assert np.allclose(out, expected)
