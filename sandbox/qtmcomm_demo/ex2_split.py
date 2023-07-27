import time

from mpi4py.MPI import COMM_WORLD
from qtm.mpi import QTMComm

# Creating 'QTMComm' instance from 'COMM_WORLD'
comm_world = QTMComm(COMM_WORLD)

if comm_world.size != 4:
    raise RuntimeError(
        "This example requires 4 MPI processes. "
        f"got COMM_WORLD.size = {comm.size}"
    )

# Processes are divided into two groups
# rank    0  1  2  3
colors = [0, 1, 1, 0]  # Group number
keys   = [1, 1, 0, 0]  # Rank in group

wrld_size = comm_world.size
wrld_rank = comm_world.rank

c = colors[wrld_rank]
k = keys[wrld_rank]

with comm_world.Split(c, k) as comm:
    grp_id = comm.subgrp_idx
    grp_size, grp_rank = comm.size, comm.rank
    print(f"process #{wrld_rank}/{wrld_size} is assigned to "
          f"subgroup #{grp_id} and its rank is "
          f"{grp_rank}/{grp_size}")