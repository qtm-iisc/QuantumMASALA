import numpy as np
from mpi4py import MPI
from mpi4py.MPI import IN_PLACE, SUM
from mpi4py_fft import DistArray

from qtm.mpi.comm import CommMod

class Test:

    shape = (10, 10)

    def __init__(self, val):
        self.buffer = np.empty(self.shape, dtype='i8')
        self.buffer[:] = val
check = None
with CommMod(MPI.COMM_WORLD) as comm:
    comm_size, comm_rank = comm.Get_size(), comm.Get_rank()

    arr = Test(comm_rank)
    if comm_rank == 2:
        print(1/0)
    val_final = (comm_size - 1) * comm_size // 2
    comm.Allreduce_sum_inplace(arr.buffer)
    check = np.all(arr.buffer == val_final)

print(f"{comm_rank}/{comm_size}: {check}")