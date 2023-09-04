import time

from mpi4py.MPI import COMM_WORLD
from qtm.mpi import QTMComm

# Creating 'QTMComm' instance from 'COMM_WORLD'
comm_world = QTMComm(COMM_WORLD)

# Function to add timestamp before printing to stdout
def print_msg(msg: str):
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"{curr_time}: {msg}")

if comm_world.size != 4:
    raise RuntimeError(
        "This example requires 4 MPI processes. "
        f"got COMM_WORLD.size = {comm.size}"
    )
    
grp_iproc = [0, 3]

with comm_world.Incl(grp_iproc) as comm:
    wrld_size, wrld_rank = comm_world.size, comm_world.rank
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