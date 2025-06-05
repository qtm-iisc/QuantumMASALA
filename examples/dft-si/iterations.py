import os
import subprocess
import time

# List of processor counts to run the process on
processor_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40]

# Number of iterations for each processor count
num_iterations = 10

# Function to run the process
def run_process(processor_count, iteration):
    directory = f"si_{processor_count}"
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, f"output_{processor_count}_{iteration}.txt")
    command = f"nohup mpirun -np {processor_count} python si_scf_supercell.py 3 > {output_file} &"
    subprocess.run(command, shell=True)
    time.sleep(1) # Ensure the process starts before moving to the next iteration

# Main loop to run the process for each processor count and iteration
for processor_count in processor_counts:
    for iteration in range(1, num_iterations + 1):
        print(f"Running on {processor_count} cores, iteration {iteration}")
        run_process(processor_count, iteration)
        # Wait for the current process to complete before starting the next one
        subprocess.run("wait", shell=True)

print("All processes have been started.")