import time

from mpi4py.MPI import COMM_WORLD
from qtm.mpi import QTMComm

# Creating 'QTMComm' instance from 'COMM_WORLD'
comm_world = QTMComm(COMM_WORLD)
wrld_size, wrld_rank = comm_world.size, comm_world.rank

# Function to add timestamp before printing to stdout
def print_msg(msg: str):
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"{curr_time}: {msg}")
    
with comm_world as comm:
    size, rank = comm.size, comm.rank
    print_msg(f"Hello from process #{rank}/{size}")
    # Proc 1 going to sleep while the rest skip ahead
    if comm.rank == 1:
        print_msg(f"process #{rank} going to sleep for 3 seconds")
        time.sleep(3)
    print_msg(f"process #{rank} is at the end of 'with' code-block")

# When exiting, all procs in comm will be in sync.
# No need for 'comm.Barrier()' here.
# So all procs will print the message below at the exact time.
print_msg(f"process #{rank} has exited 'with' code-block")

# Processes are divided into two groups
# rank    0  1  2  3
colors = [0, 1, 1, 0]  # Group number
keys   = [1, 1, 0, 0]  # Rank in group


c = colors[wrld_rank]
k = keys[wrld_rank]

with comm_world.Split(c, k) as comm:
    grp_id = comm.subgrp_idx
    grp_size, grp_rank = comm.size, comm.rank
    print(f"process #{wrld_rank}/{wrld_size} is assigned to "
          f"subgroup #{grp_id} and its rank is "
          f"{grp_rank}/{grp_size}")
    
# Selecting the processess to include in subgroup
grp_iproc = [0, 3]

with comm_world.Incl(grp_iproc) as comm:
    if not comm.is_null:
        grp_size, grp_rank = comm.size, comm.rank
        print_msg(f"process #{wrld_rank}/{wrld_size} is part of the "
                  f"subgroup and its rank is {grp_rank}/{grp_size}")
        print_msg(f"process #{wrld_rank} going to sleep for 3 seconds")
        time.sleep(3)
    else:
        print_msg(f"process #{wrld_rank}/{wrld_size} is not part of "
                  "the subgroup")
    print_msg(f"process #{wrld_rank} is at the end of 'with' code-block")

# When exiting, all procs in comm will be in sync.
# No need for 'comm.Barrier()' here.
# So all procs will print the message below at the exact time.
print_msg(f"process #{comm_world.rank} has exited 'with' code-block")