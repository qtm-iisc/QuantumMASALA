"""Simple, DFT-free unit tests for the two other `qtm.dft.eigsolve` backends
alongside 'davidson' (see '../dft_tests/test_davidson.py'): `scipy_eigsh`
(wraps `scipy.sparse.linalg.eigsh`, a core dependency) and `primme_eigsh`
(wraps the optional, not-in-pyproject `primme` package -- skipped if it
isn't installed). Both are documented in their own source files as "purely
for demonstration", so this checks they still actually work.

Uses the exact same closed-form Hamiltonian construction as
'test_davidson.py': a 'KSHam' with no nonlocal projectors and either a zero
or spatially uniform local potential, whose spectrum is known exactly
(0.5*|G+k|^2, optionally shifted by a constant) -- see that file for the
'vloc' pre-division-by-'size_r' convention this relies on.
"""
import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED, PRIMME_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

from qtm.containers import get_FieldR, get_WavefunG
from qtm.dft import DFTCommMod, KSHam, KSWfn
from qtm.dft.eigsolve import primme_eigsh, scipy_eigsh
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace
from qtm.lattice import ReciLattice
from qtm.mpi import QTMComm

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gwfn = GSpace(recilat, 20.0)
gkspc = GkSpace(gwfn, (0.13, 0.27, -0.08))  # generic k: no accidental degeneracies
FieldR = get_FieldR(gwfn)

NUMEIG = 5
DIAGO_THR = 1e-10


def _init_kswfn():
    kswfn = KSWfn(gkspc, k_weight=1.0, numbnd=NUMEIG, is_noncolin=False)
    kswfn.init_random()
    kswfn.evc_gk.normalize()
    return kswfn


def _run(name, ksham, kswfn, vloc_g0):
    if name == "scipy":
        return scipy_eigsh.solve(dftcomm, ksham, kswfn, DIAGO_THR)
    elif name == "primme":
        return primme_eigsh.solve(
            dftcomm, ksham, kswfn, DIAGO_THR, vloc_g0, numwork=8, maxiter=200
        )
    raise ValueError(name)


def _assert_solution_correct(out_wfn, ksham, exact):
    assert np.allclose(np.sort(out_wfn.evl), exact, atol=1e-8)

    evc = out_wfn.evc_gk
    assert np.allclose(evc.norm2(), 1.0, atol=1e-8)
    ovl = evc.vdot(evc)
    assert np.allclose(ovl, np.eye(NUMEIG), atol=1e-6)

    WavefunG = get_WavefunG(gkspc, 1)
    hpsi = WavefunG.empty(NUMEIG)
    ksham.h_psi(evc, hpsi)
    # Eigenvectors may come back in a different order than 'evl' if a
    # backend doesn't sort -- compare band-by-band using each band's own
    # (already-sorted) eigenvalue rather than assuming any particular order.
    residual = hpsi.data - out_wfn.evl[:, np.newaxis] * evc.data
    assert np.max(np.abs(residual)) < 1e-5


@pytest.mark.parametrize(
    "name",
    [
        "scipy",
        pytest.param(
            "primme",
            marks=pytest.mark.skipif(not PRIMME_INSTALLED, reason="primme not installed"),
        ),
    ],
)
def test_free_electron_matches_exact_kinetic_spectrum(name):
    vloc = FieldR.zeros(())
    ksham = KSHam(gkspc, is_noncolin=False, vloc=vloc, l_nloc=[])
    exact = np.sort(0.5 * gkspc.gk_norm2)[:NUMEIG]

    out_wfn, _ = _run(name, ksham, _init_kswfn(), vloc_g0=[0.0])
    _assert_solution_correct(out_wfn, ksham, exact)


@pytest.mark.parametrize(
    "name",
    [
        "scipy",
        pytest.param(
            "primme",
            marks=pytest.mark.skipif(not PRIMME_INSTALLED, reason="primme not installed"),
        ),
    ],
)
def test_uniform_potential_shifts_spectrum_by_a_known_constant(name):
    v0 = 0.37
    vloc = FieldR.zeros(())
    vloc.data[:] = v0 / gwfn.size_r
    ksham = KSHam(gkspc, is_noncolin=False, vloc=vloc, l_nloc=[])
    exact = np.sort(0.5 * gkspc.gk_norm2)[:NUMEIG] + v0

    out_wfn, _ = _run(name, ksham, _init_kswfn(), vloc_g0=[v0])
    _assert_solution_correct(out_wfn, ksham, exact)
