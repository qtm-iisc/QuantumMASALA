import numpy as np
import re
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Extract energy and temperature data from a file')
parser.add_argument('file_path', type=str, help='Path to the file')
parser.add_argument('time_step', type=int, help='Time step for the simulation')
args = parser.parse_args()
file_path = args.file_path

def extract_floats(lines):
    """
    Extract floating-point numbers in scientific notation from a line.
    
    Args:
        line (str): A line of text from the file.
    
    Returns:
        list: List of floats extracted from the line.
    """
    nums=[]
    for line in lines:
        for x in re.findall(r'[-+]?\d*\.\d+e[-+]?\d+', line):
            nums.append(float(x))

        

    return nums

def compute_vacf(initial_velocities, velocities_list):
    """
    Compute the velocity autocorrelation function for each time step.
    
    Args:
        initial_velocities (np.ndarray): Initial velocities, shape (N, 3).
        velocities_list (list): List of velocity arrays, each shape (N, 3).
    
    Returns:
        np.ndarray: VACF values for each time step.
    """
    N = initial_velocities.shape[0]  # Number of particles
    M = len(velocities_list)         # Number of time steps
    vacf = np.zeros(M)
    for t in range(M):
        # Compute dot product for each particle and average over all particles
        dot_products = np.sum(initial_velocities * velocities_list[t], axis=1)
        vacf[t] = np.mean(dot_products)
    return vacf

def parse_velocities_file(filename):
    """
    Parse an MD simulation output file to extract velocities and compute VACF.
    
    Args:
        filename (str): Path to the input file.
    
    Returns:
        np.ndarray: VACF array with length equal to the number of time steps.
    
    Raises:
        ValueError: If expected sections are not found or data is malformed.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Step 1: Extract initial velocities
    initial_marker = "re-scaled velocities in atomic units"
    for i, line in enumerate(lines):
        if initial_marker in line:
            init_vel_lines = lines[i+1:i+49]  # Next three lines (x, y, z components)
            v0_x = extract_floats(init_vel_lines[0:16])
            v0_y = extract_floats(init_vel_lines[17:32])
            v0_z = extract_floats(init_vel_lines[33:48])
            print(v0_x, len(v0_x))
            # Ensure we have 64 values per component
            if len(v0_x) != 64 or len(v0_y) != 64 or len(v0_z) != 64:
                raise ValueError("Initial velocities must have 64 values per component.")
            # Form 64x3 array by transposing
            initial_velocities = np.array([v0_x, v0_y, v0_z]).T
            break
    else:
        raise ValueError("Initial velocities marker not found in the file.")

    # Step 2: Extract velocities at each time step
    time_step_marker = "the new velocity at this iteration is"
    velocities_list = [initial_velocities]  # Include t=0
    i = 0
    while i < len(lines):
        if time_step_marker in lines[i]:
            vel_lines = lines[i+1:i+65]  # Next 64 lines
            if len(vel_lines) != 64:
                raise ValueError(f"Not enough lines for velocity data at line {i}.")
            # Parse 64 lines into a 64x3 array
            vel_array = []
            for line in vel_lines:
                components = extract_floats(line)
                if len(components) != 3:
                    raise ValueError(f"Expected 3 components per particle at line {i}.")
                vel_array.append(components)
            vel_array = np.array(vel_array)
            velocities_list.append(vel_array)
            i += 65  # Skip past the velocity block
        else:
            i += 1

    # Step 3: Compute VACF
    vacf = compute_vacf(initial_velocities, velocities_list)
    return vacf

# Example usage
# Replace 'your_file.txt' with the actual path to your MD simulation output file
# vacf = parse_velocities_file('your_file.txt')
# print("VACF:", vacf)


vacf=parse_velocities_file(file_path)
au_to_ps=0.0241888426*1e-3

time_step=args.time_step*au_to_ps
time_fs=np.arange(0,len(vacf),1)*time_step

# Plotting the VACF
plt.figure(figsize=(10, 6))
plt.plot(time_fs, vacf, label='VACF', color='blue')
plt.xlabel('Time (ps)')
plt.ylabel('VACF')
plt.title('Velocity Autocorrelation Function (VACF)')
plt.legend()
plt.savefig(f'vacf_{file_path}.png')


##Savnig theVACF
np.savetxt(f'vacf_{file_path}.txt', vacf, header='VACF values')



