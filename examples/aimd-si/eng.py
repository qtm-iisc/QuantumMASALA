import numpy as np
import re
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Extract energy and temperature data from a file')
parser.add_argument('file_path', type=str, help='Path to the file')
args = parser.parse_args()
file_path = args.file_path

# Initialize an empty list to store the energies
energies = []

# Open the file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        # Use regular expression to find energy values in both scientific notation and normal decimal numbers
        match = re.search(r'total energy of the system is ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?) eV', line)
        if match:
            # Append the energy value to the list
            energies.append(float(match.group(1)))

# Convert the list to a NumPy array
energy_array = np.array(energies)

# Initialize an empty list to store the temperatures
temperatures = []

# Open the file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        # Use regular expression to find temperature values in both scientific notation and normal decimal numbers
        match = re.search(r'The temperature of the system is ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?) K', line)
        if match:
            # Append the temperature value to the list
            temperatures.append(float(match.group(1)))

# Convert the list to a NumPy array
temperature_array = np.array(temperatures)

kin_energy=[]

# Open the file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        # Use regular expression to find temperature values in both scientific notation and normal decimal numbers
        match = re.search(r'kinetic energy of the system is ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?) eV', line)
        if match:
            # Append the temperature value to the list
            kin_energy.append(float(match.group(1)))

# Convert the list to a NumPy array
ke_array = np.array(kin_energy)

pot_energy=[]

# Open the file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        # Use regular expression to find temperature values in both scientific notation and normal decimal numbers
        match = re.search(r'potential energy of the system is ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?) eV', line)
        if match:
            # Append the temperature value to the list
            pot_energy.append(float(match.group(1)))

# Convert the list to a NumPy array
pot_array = np.array(pot_energy)

#intial totoal energy
E_tot=pot_array[0]+ke_array[0]
#renormalize energy to 0
energy_array=energy_array-E_tot
#renormalize kinetic energy to 0
#ke_array=ke_array-E_tot
#renormalize potential energy to 0
pot_array=pot_array-E_tot


length=min(len(energy_array), len(ke_array), len(pot_array))
energy_array=energy_array[:length]
ke_array=ke_array[:length]
pot_array=pot_array[:length]
time_step=np.arange(0,length,1)  
au=0.0241888426
step=20*au
time_fs=time_step*step

print(time_fs)

plt.figure(figsize=(10, 6)) 
plt.plot(time_fs, energy_array, label="total energy(renor. to 0)")
plt.plot(time_fs, ke_array, label="ke(est. E)")
plt.plot(time_fs, pot_array, label="pe")
plt.plot(time_fs, np.zeros_like(time_fs), label="zero")
plt.xlabel('Time(fs)')
plt.ylabel('Quantities(eV)')
plt.title('energy vs Time')
plt.legend()
plt.savefig(f'energy_vs_time_{file_path}.png')


##Plot only total energy 
plt.figure(figsize=(10, 6))
plt.plot(time_fs, energy_array, label="total energy(renor. to 0)")
plt.xlabel('Time(fs)')
plt.ylabel('Quantities(eV)')
plt.title('energy vs Time')
plt.legend()
plt.savefig(f'total_energy_vs_time_{file_path}.png')


