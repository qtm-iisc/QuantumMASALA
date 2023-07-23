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
    
with comm_world as comm:
    size, rank = comm.size, comm.rank
    print_msg(f"Hello from process #{rank}/{size}")
    # Proc 1 going to sleep while the rest skip ahead
    if comm.rank == 1:
        print_msg(f"process #{rank} going to sleep for 3 seconds")
        time.sleep(3)
        print_msg(f"process #{rank} woke up from sleep")
    print_msg(f"process #{rank} is at the end of 'with' code-block")

# When exiting, all procs in comm will be in sync.
# No need for 'comm.Barrier()' here.
# So all procs will print the message below at the exact time.
print_msg(f"process #{rank} has exited 'with' code-block")