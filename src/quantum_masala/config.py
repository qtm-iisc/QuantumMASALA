from importlib.util import find_spec


MPI4PY_INSTALLED = find_spec("mpi4py") is not None
CUPY_INSTALLED = find_spec("cupy") is not None
PRIMME_INSTALLED = find_spec("primme") is not None

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
    USE_MPI = COMM_WORLD.Get_size() != 1
else:
    USE_MPI = False

