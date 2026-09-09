"""Simple, DFT-free unit tests for `qtm.dft.mixing`: `MixModBase._dot`'s
symmetry/bilinearity and `compute_error`'s vanishing on identical densities,
plus the closed-form "first iteration is plain linear mixing" behavior
shared by `ModBroyden` and `GenBroyden` (there is no mixing history yet, so
both reduce to `rho_in + beta*(rho_out - rho_in)`).

Uses a plain synthetic cubic `GSpace` (as in '../gspace_tests/test_gspace.py')
and a serial `DFTCommMod`/`QTMComm` (same pattern used throughout
'tests/system_tests/'), no crystal/pseudopotential/SCF loop involved.
"""
import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

from qtm.containers import get_FieldG
from qtm.dft import DFTCommMod
from qtm.dft.mixing.genbro import GenBroyden
from qtm.dft.mixing.modbro import ModBroyden
from qtm.gspace.gspc import GSpace
from qtm.lattice import ReciLattice
from qtm.mpi import QTMComm

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gsp = GSpace(recilat, 10.0)
FieldG = get_FieldG(gsp)

rng = np.random.default_rng(0)


def _rand_rho():
    rho = FieldG.empty(1)
    rho.data[:] = rng.standard_normal((1, gsp.size_g)) + 1j * rng.standard_normal(
        (1, gsp.size_g)
    )
    return rho


# ----- MixModBase._dot / compute_error --------------------------------------
def test_dot_is_symmetric_and_bilinear():
    mm = ModBroyden(dftcomm, _rand_rho(), beta=0.5, mixdim=4)
    a, b, c = _rand_rho(), _rand_rho(), _rand_rho()

    assert np.isclose(mm._dot(a, b), mm._dot(b, a))
    assert np.isclose(mm._dot(a, b) + mm._dot(c, b), mm._dot(a + c, b))
    assert np.isclose(2.0 * mm._dot(a, b), mm._dot(2.0 * a, b))


def test_compute_error_vanishes_for_identical_densities():
    mm = ModBroyden(dftcomm, _rand_rho(), beta=0.5, mixdim=4)
    rho = _rand_rho()
    assert np.isclose(mm.compute_error(rho, rho), 0.0, atol=1e-10)


# ----- first-iteration closed form (no mixing history yet) -----------------
@pytest.mark.parametrize("MixCls", [ModBroyden, GenBroyden])
def test_first_iteration_reduces_to_plain_linear_mixing(MixCls):
    beta = 0.4
    mm = MixCls(dftcomm, _rand_rho(), beta=beta, mixdim=4)
    assert mm.idxiter == 0

    rho_in, rho_out = _rand_rho(), _rand_rho()
    expected = rho_in.data + beta * (rho_out.data - rho_in.data)

    result = mm.mix(rho_in, rho_out)
    assert np.allclose(result.data, expected, atol=1e-10)
    assert mm.idxiter == 1
