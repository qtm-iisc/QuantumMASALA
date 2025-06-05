import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to find and average times from the collated output file
def find_array(input_file):
    print(f"Processing {input_file}")
    ewald_times = []
    local_times = []
    non_local_times = []
    total_times = []
    ewald_pattern = re.compile(r'Time taken for ewald force:\s+([\d.]+)')
    local_pattern = re.compile(r'Time taken for local force:\s+([\d.]+)')
    non_local_pattern = re.compile(r'Time taken for non local force:\s+([\d.]+)')
    total_pattern = re.compile(r'Time taken for force:\s+([\d.]+)')

    with open(input_file, 'r') as file:
        for line in file:
            ewald_match = ewald_pattern.search(line)
            if ewald_match:
                ewald_times.append(float(ewald_match.group(1)))
            
            local_match = local_pattern.search(line)
            if local_match:
                local_times.append(float(local_match.group(1)))
            
            non_local_match = non_local_pattern.search(line)
            if non_local_match:
                non_local_times.append(float(non_local_match.group(1)))
            
            total_match = total_pattern.search(line)
            if total_match:
                total_times.append(float(total_match.group(1)))
    
    print("before taking mean ewald", ewald_times)
    print("before taking mean local", local_times)
    print("before taking mean non local", non_local_times)

    ewald_mean = np.mean(ewald_times)
    local_mean = np.mean(local_times)
    non_local_mean = np.mean(non_local_times)

    ewald_std = np.std(ewald_times)
    local_std = np.std(local_times)
    non_local_std = np.std(non_local_times)

    
    #total_times = np.mean(total_times)

    print(f"Ewald Time: {ewald_mean}")
    print(f"Local Time: {local_mean}")
    print(f"Non Local Time: {non_local_mean}")
    print(f"Ewald Standard Deviation: {ewald_std}")
    print(f"Local Standard Deviation: {local_std}")
    print(f"Non Local Standard Deviation: {non_local_std}")

    # Extract the processor count from the input file name
    # Extract the processor count from the input file name
    print(f"Input file: {input_file}")
    print(input_file.split('_'))
    processor_count = int(input_file.split('_')[-1].split('.')[0]) 
    return ewald_mean, local_mean, non_local_mean, processor_count, ewald_std, local_std, non_local_std

# List of directories
base_dir = "."
sub_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("si_")])

# Arrays to store the extracted times
ewald_times = []
local_times = []
non_local_times = []
total_times = []
processors = []
ewald_stds = []
local_stds = []
non_local_stds = []

# Iterate through each subdirectory and run the find_array function
for sub_dir in sub_dirs:
    for file in os.listdir(sub_dir):
        if file.startswith("collated_output_") and file.endswith(".txt"):
            collated_output_file = os.path.join(sub_dir, file)
            if os.path.isfile(collated_output_file):
                ewald, local, non_local, proc, ewald_std, local_std, non_local_std = find_array(collated_output_file)
                ewald_times.append(ewald)
                local_times.append(local)
                non_local_times.append(non_local)
                processors.append(proc)
                ewald_stds.append(ewald_std)
                local_stds.append(local_std)
                non_local_stds.append(non_local_std)


# Convert lists to numpy arrays
ewald_times = np.array(ewald_times)
local_times = np.array(local_times)
non_local_times = np.array(non_local_times)
processors = np.array(processors)
ewald_stds = np.array(ewald_stds)
local_stds = np.array(local_stds)
non_local_stds = np.array(non_local_stds)
#total_times = np.array(total_times)

print(ewald_times)
print(local_times)
print(non_local_times)

# Processor numbers
processors = np.array(processors)
sorted_indices = np.argsort(processors)
processors = processors[sorted_indices]
ewald_times = ewald_times[sorted_indices]
local_times = local_times[sorted_indices]
non_local_times = non_local_times[sorted_indices]
ewald_stds = ewald_stds[sorted_indices]
local_stds = local_stds[sorted_indices]
non_local_stds = non_local_stds[sorted_indices]

print("the standatd deviation of ewald is", ewald_stds)
print("the standatd deviation of local is", local_stds)
print("the standatd deviation of non local is", non_local_stds)


#  Plot the timings with standard deviation
plt.figure(figsize=(10, 6))
plt.errorbar(processors, ewald_times, yerr=ewald_stds, label='Ewald Force Time', marker='o', capsize=5)
plt.errorbar(processors, local_times, yerr=local_stds, label='Local Force Time', marker='o', capsize=5)
plt.errorbar(processors, non_local_times, yerr=non_local_stds, label='Non-Local Force Time', marker='o', capsize=5)
#plt.errorbar(processors, total_times, yerr=total_std, label='Total Force Time', marker='o', capsize=5)

# Add labels and title
plt.xlabel('Number of Processors')
plt.ylabel('Time (seconds)')
plt.title('Force Calculation Times vs Number of Processors')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig('force_calculation_times.png')
plt.show()


##Also plot log log plot
##Add all the markers in the x axis
plt.figure(figsize=(10, 6))
plt.loglog(processors, ewald_times, label='Ewald Force Time', marker='o')
plt.loglog(processors, local_times, label='Local Force Time', marker='o')
plt.loglog(processors, non_local_times, label='Non-Local Force Time', marker='o')

# Add labels and title
plt.xlabel('Number of Processors')
plt.ylabel('Time (seconds)')
plt.title('Force Calculation Times vs Number of Processors')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig('force_calculation_times_loglog.png')
plt.show()
