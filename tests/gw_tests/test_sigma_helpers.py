"""Simple, DFT-free unit tests for four pure G-vector-bookkeeping helper
methods of `qtm.gw.sigma.Sigma`: `rhohash`, `index_minusq`,
`map_g_to_g_minusq`, and `find_indices`.

`Sigma.__init__` needs a full converged SCF calculation's worth of crystal/
wavefunction data, so these tests build a bare `Sigma` instance via
`object.__new__` and set only the specific attributes (`.gspace.grid_shape`
or `.qpts`) each method actually touches -- no crystal, pseudopotential, or
SCF involved. `.qpts` uses the plain `QPoints` class from `qtm.gw.core`,
which itself only needs a bare `ReciLattice`.
"""
from types import SimpleNamespace

import numpy as np

from qtm.gw.core import QPoints
from qtm.gw.sigma import Sigma
from qtm.lattice import ReciLattice

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))


def _sigma():
    return object.__new__(Sigma)


# ----- rhohash -----------------------------------------------------------------
def test_rhohash_matches_manual_formula():
    sigma = _sigma()
    sigma.gspace = SimpleNamespace(grid_shape=(4, 5, 6))
    g = np.array([1, -2, 3])
    n0, n1, n2 = 4, 5, 6
    expected = (g[0] + n0 // 2) + (g[1] + n1 // 2) * n0 + (g[2] + n2 // 2) * n0 * n1
    assert sigma.rhohash(g) == expected


def test_rhohash_is_injective_over_the_full_grid():
    sigma = _sigma()
    n0, n1, n2 = 4, 5, 6
    sigma.gspace = SimpleNamespace(grid_shape=(n0, n1, n2))
    hashes = set()
    count = 0
    for x in range(-(n0 // 2), (n0 + 1) // 2):
        for y in range(-(n1 // 2), (n1 + 1) // 2):
            for z in range(-(n2 // 2), (n2 + 1) // 2):
                hashes.add(int(sigma.rhohash(np.array([x, y, z]))))
                count += 1
    assert len(hashes) == count


# ----- index_minusq --------------------------------------------------------------
def test_index_minusq_finds_the_wrapped_negative_partner():
    sigma = _sigma()
    qcryst = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma (index_q0)
            [0.25, 0.0, 0.0],
            [0.75, 0.0, 0.0],  # 1 - 0.25
            [0.3, 0.3, 0.0],
            [0.7, 0.7, 0.0],  # 1 - 0.3 in both components
        ]
    )
    sigma.qpts = QPoints(recilat, None, qcryst)
    assert sigma.qpts.index_q0 == 0

    assert sigma.index_minusq(1) == 2
    assert sigma.index_minusq(2) == 1
    assert sigma.index_minusq(3) == 4
    assert sigma.index_minusq(4) == 3


def test_index_minusq_maps_gamma_to_itself():
    sigma = _sigma()
    qcryst = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    sigma.qpts = QPoints(recilat, None, qcryst)
    assert sigma.index_minusq(0) == 0


# ----- map_g_to_g_minusq -------------------------------------------------------
def test_map_g_to_g_minusq_inverts_a_synthetic_permutation():
    sigma = _sigma()
    rng = np.random.default_rng(2)
    n = 6
    g = rng.integers(-3, 4, size=(n, 3)).astype(float)
    q = np.array([0.13, -0.27, 0.05])
    minusq = np.array([-0.13, 0.27, -0.05])

    # By construction, g_at_minusq[j] = -(g[i]+q) - minusq exactly satisfies
    # the matching condition for g[i] when j==i; shuffle so the mapping is
    # a genuine (non-identity) permutation to find.
    g_minusq_exact = -(g + q) - minusq
    perm = rng.permutation(n)
    g_at_minusq = g_minusq_exact[perm]

    indices = sigma.map_g_to_g_minusq(g, g_at_minusq, q, minusq)
    assert np.array_equal(indices, np.argsort(perm))


# ----- find_indices --------------------------------------------------------------
def test_find_indices_reports_minus_one_for_missing_targets():
    sigma = _sigma()
    targets = [(1, 2), (9, 9), (3, 4)]
    l_list = [(0, 0), (1, 2), (3, 4)]
    assert sigma.find_indices(targets, l_list) == [1, -1, 2]
