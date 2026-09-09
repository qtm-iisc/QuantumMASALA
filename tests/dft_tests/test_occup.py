"""Simple, DFT-free unit tests for `qtm.dft.occup`: `fixed.compute_occ`'s
lowest-numfill-bands filling rule and gap-edge (`max_filled`/`min_empty`)
computation, and `smear.compute_occ`'s closed-form smearing functions and
bisection-based Fermi level, checked against electron-count conservation.

Uses a `KSWfn` built on a plain synthetic cubic `GkSpace` (as in
'../gspace_tests/test_gspace.py') with hand-set `evl` arrays, and a serial
`DFTCommMod`/`QTMComm` (same pattern used throughout 'tests/system_tests/')
-- no crystal/pseudopotential/actual diagonalization needed.
"""
import numpy as np
import pytest
from scipy.special import erf

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

from qtm.dft import DFTCommMod
from qtm.dft.kswfn import KSWfn
from qtm.dft.occup import fixed as occup_fixed
from qtm.dft.occup import smear as occup_smear
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace
from qtm.lattice import ReciLattice
from qtm.mpi import QTMComm

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gwfn = GSpace(recilat, 10.0)
gkspc = GkSpace(gwfn, (0.0, 0.0, 0.0))


def _make_wfn(evl, k_weight=1.0):
    wfn = KSWfn(gkspc, k_weight, len(evl), is_noncolin=False)
    wfn.evl[:] = evl
    return wfn


# ----- occup.fixed.compute_occ -----------------------------------------------
def test_fixed_occ_fills_lowest_numfill_bands_and_finds_gap_edges():
    wfn1 = _make_wfn([0.1, 0.2, 0.3, 0.4, 0.5])
    wfn2 = _make_wfn([0.05, 0.25, 0.35, 0.45, 0.55])

    max_filled, min_empty = occup_fixed.compute_occ(dftcomm, [[wfn1], [wfn2]], numel=4)

    assert np.allclose(wfn1.occ, [1, 1, 0, 0, 0])
    assert np.allclose(wfn2.occ, [1, 1, 0, 0, 0])
    assert np.isclose(max_filled, 0.25)  # max of {0.2, 0.25}
    assert np.isclose(min_empty, 0.3)  # min of {0.3, 0.35}


def test_fixed_occ_min_empty_is_none_when_no_bands_left_over():
    wfn = _make_wfn([0.1, 0.2, 0.3])
    _, min_empty = occup_fixed.compute_occ(dftcomm, [[wfn]], numel=6)
    assert np.allclose(wfn.occ, [1, 1, 1])
    assert min_empty is None


def test_fixed_occ_rejects_odd_numel():
    # The assertion runs inside a 'with dftcomm.image_comm as comm:' block;
    # QTMComm.__exit__ re-raises exceptions from inside it as a generic
    # Exception (to keep multi-process barriers consistent) with the
    # original error chained as '__cause__', rather than propagating the
    # AssertionError directly.
    wfn = _make_wfn([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(Exception) as exc_info:
        occup_fixed.compute_occ(dftcomm, [[wfn]], numel=3)
    assert isinstance(exc_info.value.__cause__, AssertionError)


# ----- occup.smear: closed-form smearing functions --------------------------
def test_smear_functions_at_the_fermi_level_and_symmetry():
    f = np.linspace(-5.0, 5.0, 11)
    assert np.isclose(occup_smear.gauss_occ(np.array([0.0]))[0], 0.5)
    assert np.isclose(occup_smear.fd_occ(np.array([0.0]))[0], 0.5)
    # Gaussian and Fermi-Dirac smearing are symmetric about the Fermi level;
    # Marzari-Vanderbilt ("cold smearing") is deliberately NOT, and its
    # occupations legitimately overshoot 1 near the Fermi level (that's the
    # whole point of the method -- it trades that for a vanishing linear
    # entropy term), so it's checked separately below rather than assumed
    # to be bounded in [0, 1].
    assert np.allclose(occup_smear.gauss_occ(f) + occup_smear.gauss_occ(-f), 1.0)
    assert np.allclose(occup_smear.fd_occ(f) + occup_smear.fd_occ(-f), 1.0)

    g0 = -(1 / np.sqrt(2))
    expected_mv0 = 0.5 * (erf(g0) + 1) + (1 / np.sqrt(2 * np.pi)) * np.exp(-(g0**2))
    assert np.isclose(occup_smear.mv_occ(np.array([0.0]))[0], expected_mv0)


def test_compute_occ_clips_deep_tails_via_smear_threshold():
    evl = np.array([-100.0, 0.0, 100.0])  # f = -100, 0, 100 for degauss=1
    occ = occup_smear._compute_occ(evl, e_fermi=0.0, smear_typ="gauss", degauss=1.0)
    assert occ[0] == 1.0  # f << -SMEAR_THRESHOLD: hard-set to 1
    assert occ[2] == 0.0  # f >> SMEAR_THRESHOLD: never touched, stays at 0
    assert np.isclose(occ[1], 0.5)  # exactly at the Fermi level


# ----- occup.smear.compute_occ: bisection + electron-count conservation ----
@pytest.mark.parametrize("smear_typ", ["gauss", "fd", "mv"])
def test_smear_compute_occ_conserves_electron_count(smear_typ):
    wfn = _make_wfn([-1.0, -0.5, 0.0, 0.5, 1.0], k_weight=1.0)
    numel = 2.0
    occup_smear.compute_occ(
        dftcomm, [[wfn]], numel=numel, is_spin=False, smear_typ=smear_typ, degauss=0.05
    )
    assert np.isclose(np.sum(wfn.occ) * wfn.k_weight, numel, atol=1e-6)


def test_smear_compute_occ_conserves_electron_count_across_weighted_kpoints():
    wfn1 = _make_wfn([-1.0, -0.3, 0.3, 1.0], k_weight=0.5)
    wfn2 = _make_wfn([-0.9, -0.2, 0.4, 1.1], k_weight=0.5)
    numel = 2.0
    occup_smear.compute_occ(
        dftcomm, [[wfn1], [wfn2]], numel=numel, is_spin=False, smear_typ="fd", degauss=0.1
    )
    total = wfn1.k_weight * np.sum(wfn1.occ) + wfn2.k_weight * np.sum(wfn2.occ)
    assert np.isclose(total, numel, atol=1e-6)


def test_smear_compute_occ_gauss_occ_is_monotonic_non_increasing():
    # Gaussian/Fermi-Dirac occupation must decrease monotonically with
    # eigenvalue (unlike Marzari-Vanderbilt, which is not monotonic).
    wfn = _make_wfn(np.linspace(-2.0, 2.0, 9), k_weight=1.0)
    occup_smear.compute_occ(
        dftcomm, [[wfn]], numel=4.0, is_spin=False, smear_typ="gauss", degauss=0.2
    )
    assert np.all(np.diff(wfn.occ) <= 1e-12)
