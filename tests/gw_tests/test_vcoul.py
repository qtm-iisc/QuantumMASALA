"""Simple, DFT-free unit tests for `qtm.gw.vcoul.Vcoul`'s closed-form bare
and cell-averaged Coulomb interaction formulas: `v_bare` (`8*pi/|q+G|^2`,
NaN at the G=0/q=0 divergence), `v_minibz_sphere`, and
`oneoverq_minibz_sphere` (analytic averages of `8*pi/q^2` and `8*pi/q` over
a ball of the mini-BZ's volume).

Uses a plain synthetic cubic `GSpace` (as in '../gspace_tests/
test_gspace.py') and the bare `QPoints` class -- no crystal or
pseudopotential needed; `Vcoul.__init__` only needs `gspace`/`qpts`/a
cutoff.
"""
import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.gspace import GSpace
from qtm.gw.core import QPoints
from qtm.gw.vcoul import Vcoul
from qtm.lattice import ReciLattice

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gspc = GSpace(recilat, 20.0)

qcryst = np.array(
    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0], [0.05, 0.05, 0.05]]
)
qpts = QPoints(recilat, None, qcryst)
vcoul = Vcoul(gspc, qpts, bare_coulomb_cutoff=5.0, parallel=False)


def _q_cutoff():
    # The mini-BZ-equivalent-sphere radius, as computed internally by Vcoul.
    recvol = gspc.recilat.cellvol
    return (recvol / len(qpts.cryst) * 3 / (4 * np.pi)) ** (1 / 3)


def test_v_bare_matches_closed_form_away_from_the_origin():
    i_q = 1  # q = (0.1, 0, 0), no G+q == 0 in this shell
    gk = vcoul.l_gspace_q[i_q].gk_norm2
    assert not np.any(gk == 0)

    vqg = vcoul.v_bare(i_q)
    assert np.allclose(vqg, 8 * np.pi / gk)


def test_v_bare_is_nan_at_the_g0_q0_divergence():
    i_q = 0  # Gamma: G=(0,0,0) gives gk_norm2 == 0
    gk = vcoul.l_gspace_q[i_q].gk_norm2
    assert np.any(gk == 0)

    vqg = vcoul.v_bare(i_q)
    assert np.all(np.isnan(vqg[gk == 0]))
    assert np.allclose(vqg[gk != 0], 8 * np.pi / gk[gk != 0])


def test_v_minibz_sphere_matches_hand_derived_average():
    # <8*pi/r^2> over a ball of radius R:
    #   integral_0^R (8*pi/r^2)*4*pi*r^2 dr / ((4/3)*pi*R^3)
    #   = 32*pi^2*R / ((4/3)*pi*R^3) = 24*pi/R^2
    q_cutoff = _q_cutoff()
    assert np.isclose(vcoul.v_minibz_sphere(), 24 * np.pi / q_cutoff**2)


def test_oneoverq_minibz_sphere_matches_hand_derived_average():
    # <8*pi/r> over the same ball:
    #   integral_0^R (8*pi/r)*4*pi*r^2 dr / ((4/3)*pi*R^3)
    #   = 32*pi^2*R^2/2 / ((4/3)*pi*R^3) = 12*pi/R
    q_cutoff = _q_cutoff()
    assert np.isclose(vcoul.oneoverq_minibz_sphere(), 12 * np.pi / q_cutoff)


def test_oneoverq_minibz_sphere_accepts_explicit_qlen():
    assert np.isclose(vcoul.oneoverq_minibz_sphere(qlen=2.0), 12 * np.pi / 2.0)
