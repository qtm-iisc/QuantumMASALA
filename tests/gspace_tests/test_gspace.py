"""Simple, DFT-free unit tests for `qtm.gspace`: the G-vector <-> FFT-grid
indexing (`cryst2idxgrid`), the energy-cutoff truncation and canonical
sorting done by `GSpace`, the `GSpaceBase.g2r`/`r2g` FFT round trip, and the
G+k cutoff of `GkSpace`.

These check the geometric/bookkeeping machinery in isolation, against a
plain simple-cubic reciprocal lattice with no crystal or pseudopotential
involved -- 'RotateWfn' (see '../symm_tests/test_rotate_wfn.py') relies
directly on 'cryst2idxgrid' and the 'idxgrid'/'argsort'/'searchsorted'
G-vector matching exercised here, so a regression here would likely break
it too.
"""
import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.gspace.base import check_g_cryst, cryst2idxgrid
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace, minimal_grid_shape, optimal_grid_shape
from qtm.lattice import ReciLattice

# Plain simple-cubic reciprocal lattice: tpiba=1, orthonormal axes. G-vector
# norms in crystal coordinates then reduce to i1^2+i2^2+i3^2, easy to verify
# by hand, and fully decoupled from any real crystal structure.
recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))


# ----- cryst2idxgrid / check_g_cryst ---------------------------------------
def test_cryst2idxgrid_matches_manual_wrapping():
    shape = (4, 5, 6)
    g_cryst = np.array(
        [
            [0, 0, 0],
            [1, 2, 2],
            [-1, -2, -3],
            [-2, 2, -3],  # most negative/positive components allowed
        ],
        dtype="i8",
    ).T
    idx = cryst2idxgrid(shape, g_cryst)

    n1, n2, n3 = shape
    expected = [
        (i1 % n1) * n2 * n3 + (i2 % n2) * n3 + (i3 % n3) for i1, i2, i3 in g_cryst.T
    ]
    assert np.array_equal(idx, np.array(expected))


def test_cryst2idxgrid_bijective_on_full_grid():
    # Every point of a full FFT grid must map to a distinct flat index --
    # this is exactly what RotateWfn's G-vector search relies on.
    shape = (3, 4, 5)
    xi = [np.arange(-(n // 2), (n + 1) // 2) for n in shape]
    g_cryst = np.array(np.meshgrid(*xi, indexing="ij")).reshape(3, -1).astype("i8")
    idx = cryst2idxgrid(shape, g_cryst)
    assert np.array_equal(np.sort(idx), np.arange(int(np.prod(shape))))


def test_check_g_cryst_rejects_out_of_bounds():
    shape = (4, 4, 4)
    g_ok = np.array([[0], [0], [1]], dtype="i8")  # within [-2, 2)
    check_g_cryst(shape, g_ok)  # must not raise

    g_bad = np.array([[0], [0], [2]], dtype="i8")  # 2 is outside [-2, 2)
    with pytest.raises(AssertionError):
        check_g_cryst(shape, g_bad)


# ----- minimal_grid_shape / optimal_grid_shape -----------------------------
def test_minimal_grid_shape_cubic_lattice():
    # For this orthonormal reciprocal lattice, the spacing between lattice
    # planes perpendicular to each axis is exactly 1, so the formula reduces
    # to 2*floor(sqrt(2*ecut)) + 1 in every dimension.
    ecut = 10.0
    n = 2 * int(np.floor(np.sqrt(2 * ecut))) + 1
    assert minimal_grid_shape(recilat, ecut) == (n, n, n)


@pytest.mark.parametrize(
    "n,expected",
    [(1, 1), (7, 8), (8, 8), (9, 9), (10, 10), (11, 12), (13, 15), (14, 15)],
)
def test_optimal_grid_shape_hand_verified_cases(n, expected):
    assert optimal_grid_shape((n,)) == (expected,)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 23, 29, 30])
def test_optimal_grid_shape_is_5_smooth_and_not_smaller(n):
    (ni,) = optimal_grid_shape((n,))
    assert ni >= n
    x = ni
    for prime in (2, 3, 5):
        while x % prime == 0:
            x //= prime
    assert x == 1, f"{ni} (from {n}) is not 5-smooth"


# ----- GSpace: cutoff truncation, sorting, self-consistency ----------------
ecut = 10.0
gsp = GSpace(recilat, ecut)


def test_gspace_grid_shape_matches_minimal_and_optimal():
    assert gsp.grid_shape == optimal_grid_shape(minimal_grid_shape(recilat, ecut))


def test_gspace_rejects_undersized_explicit_grid_shape():
    min_shape = minimal_grid_shape(recilat, ecut)
    bad_shape = (min_shape[0] - 4,) + min_shape[1:]
    with pytest.raises(ValueError):
        GSpace(recilat, ecut, grid_shape=bad_shape)


def test_gspace_all_vectors_within_cutoff():
    assert np.all(0.5 * gsp.g_norm2 <= ecut)


def test_gspace_energy_sorted_after_undoing_grid_sort():
    # GSpace.__init__ passes G-vectors sorted by ascending energy into
    # GSpaceBase.__init__, which re-sorts them into its own canonical
    # grid-stick order for storage; 'idxsort' is the permutation that
    # recovers that original (energy-sorted) order.
    norm2_energy_order = gsp.g_norm2[gsp.idxsort]
    assert np.all(np.diff(norm2_energy_order) >= 0)


def test_gspace_g0_is_first_when_present():
    assert gsp.has_g0
    assert np.array_equal(gsp.g_cryst[:, 0], [0, 0, 0])


def test_gspace_idxgrid_self_consistent():
    # 'idxgrid' is exactly what RotateWfn's G-vector matching (idxgrid +
    # argsort + searchsorted) is built on, so this must always hold.
    assert np.array_equal(cryst2idxgrid(gsp.grid_shape, gsp.g_cryst), gsp.idxgrid)
    assert len(np.unique(gsp.idxgrid)) == gsp.size_g  # no duplicate G-vectors


def test_gspace_g2r_r2g_round_trip():
    rng = np.random.default_rng(0)
    arr_g = rng.standard_normal((3, gsp.size_g)) + 1j * rng.standard_normal(
        (3, gsp.size_g)
    )
    arr_r = gsp.g2r(arr_g)
    arr_g_back = gsp.r2g(arr_r)
    assert np.allclose(arr_g, arr_g_back, atol=1e-10)


# ----- GkSpace: G+k cutoff --------------------------------------------------
gwfn = GSpace(recilat, 20.0)


def test_gkspace_at_gamma_with_full_ecut_recovers_gwfn():
    # k=0 with ecutwfn=gwfn.ecut: every retained gwfn G-vector already
    # satisfies its own cutoff, so nothing should be dropped.
    gkspc = GkSpace(gwfn, (0.0, 0.0, 0.0), ecutwfn=gwfn.ecut)
    assert gkspc.size_g == gwfn.size_g
    assert np.array_equal(gkspc.g_cryst, gwfn.g_cryst)


def test_gkspace_selection_matches_manual_cutoff():
    k_cryst = (0.3, -0.1, 0.2)
    gkspc = GkSpace(gwfn, k_cryst)
    assert gkspc.ecutwfn == gwfn.ecut / 4

    gk = gwfn.g_cryst.astype("f8") + np.array(k_cryst)[:, np.newaxis]
    manual_mask = 0.5 * recilat.norm2(gk, "cryst") <= gkspc.ecutwfn
    expected = set(map(tuple, gwfn.g_cryst[:, manual_mask].T))
    actual = set(map(tuple, gkspc.g_cryst.T))
    assert expected == actual

    assert np.allclose(
        gkspc.gk_cryst, gkspc.g_cryst + np.array(k_cryst)[:, np.newaxis]
    )
