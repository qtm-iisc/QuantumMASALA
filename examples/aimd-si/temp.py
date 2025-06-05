import numpy as np
import re
import matplotlib.pyplot as plt

# Initialize an empty list to store the temperatures
temperatures = []

# Open the file and read line by line
with open('06_md.out', 'r') as file:
    for line in file:
        # Use regular expression to find temperature values in both scientific notation and normal decimal numbers
        match = re.search(r'The temperature of the system is ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?) K', line)
        if match:
            # Append the temperature value to the list
            temperatures.append(float(match.group(1)))

# Convert the list to a NumPy array
temperature_array = np.array(temperatures)

plt.figure(figsize=(10, 6)) 
plt.plot(temperature_array)
plt.xlabel('Time step')
plt.ylabel('Temperature (K)')
plt.title('Temperature vs Time')
plt.savefig('Temperature_vs_Time_3.png')


# Print the NumPy array
print(temperature_array)