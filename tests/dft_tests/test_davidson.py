"""Simple, DFT-free unit tests for `qtm.dft.eigsolve.davidson.solve`: the
block-Davidson generalized eigensolver used by SCF, checked against a
Hamiltonian whose exact spectrum is known in closed form.

`qtm.dft.ksham.KSHam` can be built with an empty list of nonlocal
projectors (`l_nloc=[]`) and a trivial local potential, reducing `h_psi` to
just the plane-wave kinetic energy (plus, in the second test, a uniform
potential shift) -- both have an exact, hand-computable spectrum, with no
crystal, pseudopotential, or SCF loop needed. Uses a plain synthetic cubic
`GkSpace` at a generic (non-symmetric) k-point (as in
'../gspace_tests/test_gspace.py') so the lowest eigenvalues are
non-degenerate and unambiguous to compare against.
"""
import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

from qtm.containers import get_FieldR, get_WavefunG
from qtm.dft import DFTCommMod, KSHam, KSWfn
from qtm.dft.eigsolve.davidson import solve
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


def test_davidson_free_electron_matches_exact_kinetic_spectrum():
    # With vloc=0 and no nonlocal projectors, h_psi is exactly the plane-wave
    # kinetic energy operator: diagonal in G, with eigenvalues 0.5*|G+k|^2.
    vloc = FieldR.zeros(())
    ksham = KSHam(gkspc, is_noncolin=False, vloc=vloc, l_nloc=[])

    exact = np.sort(0.5 * gkspc.gk_norm2)[:NUMEIG]
    assert exact[-1] < np.sort(0.5 * gkspc.gk_norm2)[NUMEIG] - 1e-6  # no boundary degeneracy

    kswfn = _init_kswfn()
    out_wfn, idxiter = solve(
        dftcomm, ksham, kswfn, DIAGO_THR, vloc_g0=[0.0], numwork=8, maxiter=200
    )
    assert idxiter < 200  # actually converged, didn't just hit maxiter
    assert np.allclose(out_wfn.evl, exact, atol=1e-8)

    # Eigenvectors: normalized, mutually orthogonal, and genuine eigenvectors
    # of h_psi (H|psi> = E|psi>), not just eigenvalues matching by accident.
    evc = out_wfn.evc_gk
    assert np.allclose(evc.norm2(), 1.0, atol=1e-10)
    ovl = evc.vdot(evc)
    assert np.allclose(ovl, np.eye(NUMEIG), atol=1e-6)

    WavefunG = get_WavefunG(gkspc, 1)
    hpsi = WavefunG.empty(NUMEIG)
    ksham.h_psi(evc, hpsi)
    residual = hpsi.data - out_wfn.evl[:, np.newaxis] * evc.data
    assert np.max(np.abs(residual)) < 1e-5


def test_davidson_uniform_potential_shifts_spectrum_by_a_known_constant():
    # Adding a spatially uniform potential V(r) = v0 to the Hamiltonian must
    # shift every eigenvalue by exactly v0, leaving the (plane-wave)
    # eigenvectors unchanged.
    #
    # Convention note (verified against qtm.dft.scf's actual construction of
    # 'vloc'/'vloc_g0', qtm/dft/scf.py around 'compute_vloc'): 'KSHam.vloc'
    # is expected to hold V(r) PRE-DIVIDED by the real-space grid size
    # ('gwfn.size_r' = np.prod(grid_shape)) -- this exactly compensates for
    # WavefunG/WavefunR's to_r()/to_g() round trip scaling by 'size_r' (see
    # '../containers_tests/test_containers.py::
    # test_wavefun_round_trip_scales_by_size_r'), which 'h_psi' relies on
    # internally when applying the local potential in real space. 'vloc_g0'
    # itself is NOT divided by 'size_r' -- it is the genuine physical
    # G=0/mean value of the potential, used only as a preconditioner hint.
    v0 = 0.37
    vloc = FieldR.zeros(())
    vloc.data[:] = v0 / gwfn.size_r
    ksham = KSHam(gkspc, is_noncolin=False, vloc=vloc, l_nloc=[])

    exact = np.sort(0.5 * gkspc.gk_norm2)[:NUMEIG] + v0

    kswfn = _init_kswfn()
    out_wfn, idxiter = solve(
        dftcomm, ksham, kswfn, DIAGO_THR, vloc_g0=[v0], numwork=8, maxiter=200
    )
    assert idxiter < 200
    assert np.allclose(out_wfn.evl, exact, atol=1e-8)
