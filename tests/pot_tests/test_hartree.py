"""Simple, DFT-free unit tests for `qtm.pot.hartree.compute`: the closed-form
Hartree potential `v_hartree(G) = 4*pi*rho(G)/|G|^2` (with the G=0 term
zeroed) and its interaction energy, checked directly against an
independently-derived reference on a synthetic charge density.

Uses a plain synthetic cubic `GSpace` (as in '../gspace_tests/test_gspace.py')
-- no crystal, no pseudopotential, no SCF.
"""
import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import FPI
from qtm.containers import get_FieldG
from qtm.gspace.gspc import GSpace
from qtm.lattice import ReciLattice
from qtm.pot import hartree

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gsp = GSpace(recilat, 10.0)
FieldG = get_FieldG(gsp)

rng = np.random.default_rng(0)


def _manual_hartree(chden_g):
    """chden_g: (size_g,) total (summed-over-spin) charge density in G-space."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v_g = FPI * chden_g / gsp.g_norm2
    v_g[0] = 0.0  # G=0 is excluded (G=0 is always first, per GSpace's sort)
    v_r = gsp.g2r(v_g.reshape(1, -1)).reshape(-1)
    en = (
        0.5
        * np.sum(chden_g.conj() * v_g).real
        * gsp.reallat_dv
        / np.prod(gsp.grid_shape)
    )
    return v_r, en


def _rand_rho(numspin):
    data = rng.standard_normal((numspin, gsp.size_g)) + 1j * rng.standard_normal(
        (numspin, gsp.size_g)
    )
    return FieldG.from_array(data), data


def test_hartree_matches_closed_form_nonspin():
    rho, data = _rand_rho(1)
    v_r, en = hartree.compute(rho)
    v_manual, en_manual = _manual_hartree(data[0])
    assert np.allclose(v_r.data.reshape(-1), v_manual, atol=1e-9)
    assert np.isclose(en, en_manual)


def test_hartree_matches_closed_form_spin_polarized():
    # hartree.compute sums over the spin axis before building v_hartree --
    # it depends only on the total charge density, not the spin density.
    rho, data = _rand_rho(2)
    v_r, en = hartree.compute(rho)
    v_manual, en_manual = _manual_hartree(data[0] + data[1])
    assert np.allclose(v_r.data.reshape(-1), v_manual, atol=1e-9)
    assert np.isclose(en, en_manual)


def test_hartree_potential_linear_and_energy_quadratic_in_density():
    rho, data = _rand_rho(1)
    v1, e1 = hartree.compute(rho)
    c = 2.5
    v2, e2 = hartree.compute(FieldG.from_array(c * data))
    assert np.allclose(v2.data, c * v1.data, atol=1e-9)
    assert np.isclose(e2, c**2 * e1)
