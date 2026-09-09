"""A more sophisticated (non-diagonal) ground-truth test for
`qtm.dft.eigsolve`'s three backends (`davidson`, `scipy_eigsh`,
`primme_eigsh`), going beyond the pure-kinetic/uniform-potential diagonal
cases in 'test_davidson.py'/'test_eigsolve_backends.py'.

Builds a genuine, realistic Kohn-Sham Hamiltonian for diamond Si -- the bare
ionic local pseudopotential (`qtm.pseudo.loc.loc_generate_pot_rhocore`) plus
Kleinman-Bylander nonlocal projectors (`qtm.pseudo.NonlocGenerator`), using
the actual UPF file already used elsewhere in this test suite -- with NO
Hartree/XC/SCF at all, so this is a fixed, density-independent, non-diagonal
operator. Ground truth comes from an independent DENSE diagonalization:
`ksham.h_psi` is applied to the identity (every basis unit vector, in a
single batched call) to build its explicit matrix, which is then
diagonalized with `scipy.linalg.eigh` -- a completely different code path
from any of the three iterative solvers under test. This also directly
verifies `h_psi` itself produces a genuinely Hermitian operator, which is
what makes comparing to a plain (non-generalized) dense Hermitian
eigensolver valid in the first place (the plane-wave G-space basis is
already orthonormal, so there is no overlap matrix to worry about).

'ecut' is kept modest (a bit over a hundred plane waves) so the
O(size_g^2) dense reference stays cheap, while still comfortably exceeding
'numwork*NUMEIG' -- the Davidson/primme subspace can never legitimately
exceed the total plane-wave basis dimension, which is what a too-small
'gkspc' (relative to 'numwork') ran into originally.
"""
import os

import numpy as np
import pytest
import scipy.linalg as sla

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED, PRIMME_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

from qtm.containers import get_WavefunG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, KSHam, KSWfn
from qtm.dft.eigsolve import davidson, primme_eigsh, scipy_eigsh
from qtm.gspace import GkSpace, GSpace
from qtm.lattice import RealLattice
from qtm.mpi import QTMComm
from qtm.pseudo import NonlocGenerator, UPFv2Data
from qtm.pseudo.loc import loc_generate_pot_rhocore

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

reallat = RealLattice.from_alat(
    alat=10.2, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]
)
si_oncv = UPFv2Data.from_file(
    os.path.join(
        os.path.dirname(__file__), "..", "system_tests", "Si_ONCV_PBE-1.2.upf"
    )
)
si_atoms = BasisAtoms(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]).T,
)
crystal = Crystal(reallat, [si_atoms])

gwfn = GSpace(crystal.recilat, 16.0)
gkspc = GkSpace(gwfn, (0.13, 0.27, -0.08))  # generic k: no accidental degeneracies

v_ion, _rho_core = loc_generate_pot_rhocore(si_atoms, gwfn)
vloc = v_ion.to_r() / gwfn.size_r  # see test_davidson.py for this convention
nloc = NonlocGenerator(si_atoms, gwfn)
ksham = KSHam(gkspc, is_noncolin=False, vloc=vloc, l_nloc=[nloc])
VLOC_G0 = [float(np.mean(v_ion.to_r().data).real)]

WavefunG = get_WavefunG(gkspc, 1)
SIZE_G = gkspc.size_g

# ----- Dense ground truth: apply h_psi to the identity, diagonalize -------
_ident = WavefunG.empty(SIZE_G)
_ident.data[:] = np.eye(SIZE_G)
_hpsi = WavefunG.empty(SIZE_G)
ksham.h_psi(_ident, _hpsi)
H_DENSE = _hpsi.data.T  # H_DENSE[:, j] = h_psi(e_j)

NUMEIG = 5
DIAGO_THR = 1e-10


def test_h_psi_produces_a_hermitian_matrix():
    assert np.max(np.abs(H_DENSE - H_DENSE.conj().T)) < 1e-10


def test_dense_reference_has_no_near_degeneracy_at_the_numeig_boundary():
    evl_dense = sla.eigvalsh(H_DENSE)
    assert evl_dense[NUMEIG] - evl_dense[NUMEIG - 1] > 1e-3


EXACT = sla.eigvalsh(H_DENSE)[:NUMEIG]


def _init_kswfn():
    kswfn = KSWfn(gkspc, k_weight=1.0, numbnd=NUMEIG, is_noncolin=False)
    kswfn.init_random()
    kswfn.evc_gk.normalize()
    return kswfn


def _run(name):
    kswfn = _init_kswfn()
    # numwork*NUMEIG must stay comfortably below SIZE_G, since the Davidson/
    # primme subspace can never legitimately exceed the total plane-wave
    # basis dimension.
    if name == "davidson":
        return davidson.solve(
            dftcomm, ksham, kswfn, DIAGO_THR, VLOC_G0, numwork=8, maxiter=300
        )
    elif name == "scipy":
        return scipy_eigsh.solve(dftcomm, ksham, kswfn, DIAGO_THR)
    elif name == "primme":
        return primme_eigsh.solve(
            dftcomm, ksham, kswfn, DIAGO_THR, VLOC_G0, numwork=8, maxiter=300
        )
    raise ValueError(name)


@pytest.mark.parametrize(
    "name",
    [
        "davidson",
        "scipy",
        pytest.param(
            "primme",
            marks=pytest.mark.skipif(not PRIMME_INSTALLED, reason="primme not installed"),
        ),
    ],
)
def test_eigsolve_matches_dense_reference_on_real_pseudopotential_hamiltonian(name):
    out_wfn, _ = _run(name)
    assert np.allclose(np.sort(out_wfn.evl), EXACT, atol=1e-6)

    evc = out_wfn.evc_gk
    assert np.allclose(evc.norm2(), 1.0, atol=1e-8)
    assert np.allclose(evc.vdot(evc), np.eye(NUMEIG), atol=1e-6)

    hpsi = WavefunG.empty(NUMEIG)
    ksham.h_psi(evc, hpsi)
    residual = hpsi.data - out_wfn.evl[:, np.newaxis] * evc.data
    assert np.max(np.abs(residual)) < 1e-4
