
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_num(file_path, string):
    capture=False
    # Regular expression to match both scientific notation and decimal numbers
    number_pattern = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'

    with open(file_path, 'r') as file:
        for line in file:
            if string in line.lower():  # Detect the start of velocities section
                numbers = re.findall(number_pattern, line)
    print("numbers are", numbers)
    return numbers
# Example usage

import argparse

parser = argparse.ArgumentParser(description='Extract numbers from a file')
parser.add_argument('file_path', type=str, help='Path to the file')
args = parser.parse_args()

string="The temperature of the system is "
energy_string="The total energy of the system is "

file_path=args.file_path
print("file path is", file_path)

Temperature=extract_num(file_path, string)
x_temp=np.arange(0, len(Temperature), 1)
Energy=extract_num(file_path, energy_string)
x_energy=np.arange(0, len(Energy), 1)



plt.figure(figsize=(10, 6)) 
plt.plot(x_temp, Temperature, label='Temperature')
plt.xlabel('Time step')
plt.ylabel('Temperature')
plt.title(f'Temperature vs Time')
plt.legend()
plt.savefig(f'Temperature_vs_Time_{file_path}.png')


plt.figure(figsize=(10, 6))
plt.plot(x_energy, Energy, label='Energy')
plt.xlabel('Time step')
plt.ylabel('Energy')
plt.title('Energy vs Time')
plt.legend()
plt.savefig('Energy_vs_Time.png')
