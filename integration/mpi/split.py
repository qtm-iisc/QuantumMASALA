
from mpi4py.MPI import COMM_WORLD
from qtm.mpi.comm import QTMComm, split_comm_pwgrp


pwgrp_size = 3


world_size, world_rank = COMM_WORLD.Get_size(), COMM_WORLD.Get_rank()
if COMM_WORLD.Get_size() % pwgrp_size != 0:
    raise ValueError("# of MPI processes not compatible")
    exit()

comm_world = QTMComm(COMM_WORLD)

n_pwgrp = world_size // pwgrp_size
pwgrp_comm, intercomm = split_comm_pwgrp(comm_world, pwgrp_size)

print(f"{world_rank}/{world_size}: "
      f"PWGRP #{pwgrp_comm.subgrp_idx}/{n_pwgrp} "
      f"intracomm rank {pwgrp_comm.rank}/{pwgrp_size} - "
      f"intercomm rank {intercomm.rank}/{intercomm.size}")

