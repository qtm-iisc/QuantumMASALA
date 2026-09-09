"""Simple, DFT-free unit tests for `qtm.lattice`: coordinate-transform round
trips, the cryst/cart consistency of `dot`/`norm2`, unit-conversion
classmethods (`from_alat`/`from_bohr`/`from_angstrom`/`from_tpiba`/
`from_cart`), the real<->reciprocal lattice duality (`from_reallat`/
`from_recilat`), `get_mesh_coords`, and `Lattice.__eq__`.

Everything here operates on bare `RealLattice`/`ReciLattice` objects with no
crystal, pseudopotential, or SCF involved.
"""
import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import ANGSTROM, TPI
from qtm.lattice import Lattice, RealLattice, ReciLattice

# A generic, non-orthogonal (triclinic-like) real lattice: round trips and
# cryst/cart cross-checks are only meaningful stress tests if off-diagonal
# metric terms are actually present.
reallat = RealLattice.from_alat(
    alat=7.5, a1=[1.0, 0.0, 0.0], a2=[0.3, 1.2, 0.0], a3=[0.1, 0.2, 0.9]
)
recilat = ReciLattice.from_reallat(reallat)

rng = np.random.default_rng(0)


# ----- cart2cryst / cryst2cart round trips ---------------------------------
@pytest.mark.parametrize("lat", [reallat, recilat])
def test_cart2cryst_cryst2cart_round_trip(lat):
    v_cart = rng.standard_normal((3, 5))
    v_cryst = lat.cart2cryst(v_cart)
    assert np.allclose(lat.cryst2cart(v_cryst), v_cart, atol=1e-10)

    v_cryst2 = rng.standard_normal((3, 5))
    v_cart2 = lat.cryst2cart(v_cryst2)
    assert np.allclose(lat.cart2cryst(v_cart2), v_cryst2, atol=1e-10)


# ----- dot / norm2 cryst vs cart consistency --------------------------------
@pytest.mark.parametrize("lat", [reallat, recilat])
def test_dot_and_norm2_consistent_across_coords(lat):
    v1_cryst = rng.standard_normal((3, 4))
    v2_cryst = rng.standard_normal((3, 4))
    v1_cart = lat.cryst2cart(v1_cryst)
    v2_cart = lat.cryst2cart(v2_cryst)

    dot_cryst = np.diagonal(lat.dot(v1_cryst, v2_cryst, "cryst"))
    dot_cart = np.diagonal(lat.dot(v1_cart, v2_cart, "cart"))
    assert np.allclose(dot_cryst, dot_cart, atol=1e-8)

    norm2_cryst = lat.norm2(v1_cryst, "cryst")
    norm2_cart = lat.norm2(v1_cart, "cart")
    assert np.allclose(norm2_cryst, norm2_cart, atol=1e-8)
    assert np.allclose(lat.norm(v1_cryst, "cryst"), np.sqrt(norm2_cryst))


# ----- metric tensor / cell volume ------------------------------------------
def test_metric_is_gram_matrix_of_primvec():
    assert np.allclose(reallat.metric, reallat.primvec.T @ reallat.primvec)
    # Symmetric positive-definite, since it's a Gram matrix of independent vectors
    assert np.allclose(reallat.metric, reallat.metric.T)
    assert np.all(np.linalg.eigvalsh(reallat.metric) > 0)


def test_cellvol_matches_determinant():
    assert np.isclose(reallat.cellvol, np.linalg.det(reallat.primvec))
    assert reallat.cellvol > 0  # right-handed axes as constructed above


# ----- real <-> reciprocal duality ------------------------------------------
def test_real_reciprocal_duality():
    # b_i . a_j = 2*pi*delta_ij, by definition of the dual lattice
    assert np.allclose(recilat.recvec.T @ reallat.latvec, TPI * np.eye(3))


def test_from_recilat_from_reallat_round_trip():
    reallat_back = RealLattice.from_recilat(recilat)
    assert reallat_back == reallat  # exercises the alat-aware __eq__ fix

    recilat_back = ReciLattice.from_reallat(reallat_back)
    assert recilat_back == recilat  # exercises the tpiba-aware __eq__ fix


# ----- Lattice.__eq__ must account for the scale parameter -----------------
def test_eq_distinguishes_alat_and_tpiba_even_with_equal_matrix():
    a = RealLattice.from_alat(alat=2.0, a1=[1.0, 0, 0], a2=[0, 1.0, 0], a3=[0, 0, 1.0])
    b = RealLattice.from_bohr(alat=5.0, a1=[2.0, 0, 0], a2=[0, 2.0, 0], a3=[0, 0, 2.0])
    assert np.allclose(a.latvec, b.latvec)  # same matrix ...
    assert a.alat != b.alat  # ... but a different 'alat' label
    assert a != b

    c = RealLattice.from_alat(alat=2.0, a1=[1.0, 0, 0], a2=[0, 1.0, 0], a3=[0, 0, 1.0])
    assert a == c  # matrix and alat both match

    p = ReciLattice.from_tpiba(1.0, (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
    q = ReciLattice.from_cart(3.0, (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
    assert np.allclose(p.recvec, q.recvec)
    assert p.tpiba != q.tpiba
    assert p != q


# ----- unit-conversion classmethods -----------------------------------------
def test_from_alat_scales_axes_by_alat():
    alat, a1, a2, a3 = 3.3, [1.0, 0.2, 0], [0, 1.0, 0.1], [0.1, 0, 1.0]
    lat = RealLattice.from_alat(alat, a1, a2, a3)
    assert np.allclose(lat.latvec, alat * np.stack([a1, a2, a3], axis=1))
    # axes_alat should recover exactly the (unscaled) input directions
    assert np.allclose(np.array(lat.axes_alat).T, np.stack([a1, a2, a3], axis=1))


def test_from_bohr_and_from_angstrom_are_unit_consistent():
    a1, a2, a3 = [2.0, 0, 0], [0, 3.0, 0], [0, 0, 4.0]
    lat_bohr = RealLattice.from_bohr(
        alat=ANGSTROM, a1=[x * ANGSTROM for x in a1],
        a2=[x * ANGSTROM for x in a2], a3=[x * ANGSTROM for x in a3],
    )
    lat_ang = RealLattice.from_angstrom(alat=1.0, a1=a1, a2=a2, a3=a3)
    assert np.isclose(lat_bohr.alat, lat_ang.alat)
    assert np.allclose(lat_bohr.latvec, lat_ang.latvec)


def test_from_tpiba_scales_but_from_cart_does_not():
    b1, b2, b3 = (1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)
    tpiba = 2.5

    lat_tpiba = ReciLattice.from_tpiba(tpiba, b1, b2, b3)
    assert np.allclose(lat_tpiba.recvec, tpiba * np.stack([b1, b2, b3], axis=1))

    lat_cart = ReciLattice.from_cart(tpiba, b1, b2, b3)
    assert np.allclose(lat_cart.recvec, np.stack([b1, b2, b3], axis=1))  # unscaled


# ----- get_mesh_coords -------------------------------------------------------
def test_get_mesh_coords_wraps_into_unit_cell_and_is_a_bijection():
    n1, n2, n3 = 3, 4, 5
    mesh = reallat.get_mesh_coords(n1, n2, n3, coords="cryst")
    assert mesh.shape == (3, n1, n2, n3)
    # Wrapping uses '-= np.rint(...)', i.e. round-half-to-even: for an even
    # n, the fraction 0.5 rounds to 0 (not 1) and so stays at +0.5 rather
    # than wrapping to -0.5 -- the cell is [-0.5, 0.5], closed at +0.5 for
    # even n, not the half-open [-0.5, 0.5) one might naively expect.
    assert np.all(mesh >= -0.5) and np.all(mesh <= 0.5)

    for axis, n in enumerate((n1, n2, n3)):
        # mesh[axis]*n mod n must realize every residue 0..n-1 (a bijection
        # onto the n grid points along that axis, just wrapped to [-0.5,0.5)).
        residues = np.rint(mesh[axis] * n).astype("i8") % n
        assert set(np.unique(residues)) == set(range(n))


def test_get_mesh_coords_origin_is_a_pure_translation():
    n1, n2, n3 = 3, 3, 3
    origin = (0.4, -0.1, 0.25)
    mesh0 = reallat.get_mesh_coords(n1, n2, n3, coords="cryst")
    mesh_shifted = reallat.get_mesh_coords(n1, n2, n3, coords="cryst", origin=origin)

    diff = mesh_shifted - mesh0 + np.array(origin).reshape(3, 1, 1, 1)
    assert np.allclose(diff - np.rint(diff), 0.0, atol=1e-10)
