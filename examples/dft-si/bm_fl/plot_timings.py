import re
import argparse
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser(description='Extract times from collated output file.')
parser.add_argument('input_file', type=str, help='Path to the collated output file')
args = parser.parse_args()

# File containing the collated output
input_file = args.input_file

# Arrays to store the extracted times
ewald_times = []
local_times = []
non_local_times = []
total_times = []

# Regular expressions to match the lines containing the times
ewald_pattern = re.compile(r'Time taken for ewald force:\s+([\d.]+)')
local_pattern = re.compile(r'Time taken for local force:\s+([\d.]+)')
non_local_pattern = re.compile(r'Time taken for non local force:\s+([\d.]+)')
total_pattern = re.compile(r'Time taken for force:\s+([\d.]+)')

# Read the input file and extract the times
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

# Processor numbers
processors = [1,2,3,4,5,6,7,8,9,10, 12, 15, 20,25,30,40]

# Plot the timings
plt.figure(figsize=(10, 6))
plt.plot(processors, ewald_times, label='Ewald Force Time', marker='o')
plt.plot(processors, local_times, label='Local Force Time', marker='o')
plt.plot(processors, non_local_times, label='Non-Local Force Time', marker='o')
plt.plot(processors, total_times, label='Total Force Time', marker='o')

# Add labels and title
plt.xlabel('Number of Processors')
plt.ylabel('Time (seconds)')
plt.title('Force Calculation Times vs Number of Processors')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig('force_calculation_times.png')
plt.show()
