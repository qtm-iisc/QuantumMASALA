"""Driver (not collected by pytest -- see 'test_*.py' naming convention):
run under `mpirun -np N python3 pwgrp_fft_roundtrip.py` by
`tests/mpi_tests/test_dist_pwgrp.py`.

Checks that `DistGSpace`'s distributed pencil-FFT (`to_r()`/`to_g()`) is
exact to machine precision at N ranks, by comparing a round trip through it
against a round trip through a plain (serial) `GSpace` on the SAME random
G-space vector. Exits with an assertion failure (nonzero exit code) if not.
"""
import numpy as np

from qtm.config import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import RYDBERG
from qtm.containers.field import get_FieldG
from qtm.gspace import GSpace
from qtm.lattice import ReciLattice
from qtm.mpi import QTMComm
from qtm.mpi.gspace import DistGSpace
from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gspc_serial = GSpace(recilat, 10 * RYDBERG)

np.random.seed(0)
g_glob = None
if comm_world.rank == 0:
    g_glob = (
        np.random.randn(1, gspc_serial.size_g)
        + 1j * np.random.randn(1, gspc_serial.size_g)
    ).astype("c16")
g_glob = comm_world.bcast(g_glob)

FieldG_serial = get_FieldG(gspc_serial)
f_serial = FieldG_serial.zeros((1,))
f_serial.data[:] = g_glob
f_g_serial_roundtrip = f_serial.to_r().to_g().data

if comm_world.size == 1:
    g_dist_glob = f_g_serial_roundtrip
else:
    gspc_dist = DistGSpace(comm_world, gspc_serial)
    FieldG_dist = get_FieldG(gspc_dist)
    f_dist = FieldG_dist.zeros((1,))
    f_dist.data[:] = gspc_dist.scatter_g(g_glob if comm_world.rank == 0 else None)
    f_g_dist_roundtrip = f_dist.to_r().to_g()
    g_dist_glob = gspc_dist.allgather_g(f_g_dist_roundtrip.data)

if comm_world.rank == 0:
    err = np.max(np.abs(g_dist_glob - f_g_serial_roundtrip))
    assert err < 1e-10, f"distributed FFT round trip diverged from serial: {err:.3e}"
    print(f"OK: np={comm_world.size}, roundtrip err vs serial = {err:.3e}")
