import numpy as np
import re

def extract_velocities(file_path):
    velocities = []
    capture = False  # Flag to track when we are inside the velocity section

    # Regular expression to match both scientific notation and decimal numbers
    number_pattern = r'[-+]?\d*\.\d+(?:e[-+]?\d+)?'

    with open(file_path, 'r') as file:
        for line in file:
            if "re-scaled velocities" in line.lower():  # Detect the start of velocities section
                capture = True
                continue  # Skip this line
            
            if capture:
                # Check if the line contains numbers; stop if it's unrelated data
                numbers = re.findall(number_pattern, line)
                
                if not numbers:  # Stop capturing if no numbers are found
                    break  
                
                # Convert extracted strings to float
                velocities.append([float(x) for x in numbers])

    velocities=np.array(velocities)
    vel_input = ""
    for coord in velocities:
        ##truncate the coordinates to 8 decimal places
        coord = [round(c, 9) for c in coord]
        vel_input += f"Si {coord[0]} {coord[1]} {coord[2]}\n"

    return vel_input

def extract_coordinates(file_path):
    coordinates = []
    capture = False  # Flag to track when we are inside the velocity section

    # Regular expression to match both scientific notation and decimal numbers
    number_pattern = r'[-+]?\d*\.\d+(?:e[-+]?\d+)?'

    with open(file_path, 'r') as file:
        for line in file:
            if "the original coordinates are" in line.lower():  # Detect the start of velocities section
                capture = True
                continue  # Skip this line
            
            if capture:
                # Check if the line contains numbers; stop if it's unrelated data
                numbers = re.findall(number_pattern, line)
                
                if not numbers:  # Stop capturing if no numbers are found
                    break  
                
                # Convert extracted strings to float
                coordinates.append([float(x) for x in numbers])

    coord_input = ""   
    coordinates=np.array(coordinates) 
    for coord in coordinates:
        ##truncate the coordinates to 8 decimal places
        coord = [round(c, 9) for c in coord]
        coord_input += f"Si {coord[0]} {coord[1]} {coord[2]}\n"
    num_atoms=coordinates.shape[0]
    supercell_size=int((num_atoms/2)**(1/3))
    return coord_input, num_atoms, supercell_size

def extract_num(file_path, string):
    capture=False
    # Regular expression to match both scientific notation and decimal numbers
    number_pattern = r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'

    with open(file_path, 'r') as file:
        for line in file:
            if string in line.lower():  # Detect the start of velocities section
                numbers = re.findall(number_pattern, line)
    print("numbers are", numbers)
    return numbers[0]
# Example usage

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="Path to the file containing velocities")
parser.add_argument("output", help="Path to the file where the velocities will be written")
args = parser.parse_args()

file_path = args.file_path
velocities_array = extract_velocities(file_path)
coordinates_array, num_atoms, supercell_size = extract_coordinates(file_path)

dt=int(extract_num(file_path, "dt="))
nstep=int(extract_num(file_path, "nstep="))
ecutwfc=float(extract_num(file_path, "ecutwfc="))
ecutwfc*=2
occupations=extract_num(file_path, "occ_typ=")
occupation="smearing" if occupations=="1" else "fixed"
smear_type=extract_num(file_path, "smear_typ=")
smear="gaussian" if smear_type=="1" else "fermi-dirac"
e_temp=float(extract_num(file_path, "e_temp="))
e_temp*=2
conv_thr=float(extract_num(file_path, "conv_thr="))
conv_thr*=2
diago_thr_init=float(extract_num(file_path, "diago_thr_init="))
mixing_beta=float(extract_num(file_path, "mixing_beta="))
numbnd=int(extract_num(file_path, "numbnd="))
T_init=float(extract_num(file_path, "initialtemp"))


input_template=f"""
    &control
    calculation = 'md'
    restart_mode='from_scratch',
    prefix='silicon',
    tstress = .true.
    tprnfor = .true.
    verbosity='high'
    pseudo_dir = '/home/sanmitc/qe-7.3/pseudo/',
    outdir='/home/sanmitc/qe-7.3/tempdir/'
    dt           = {dt}
    nstep        = {nstep}
 /
 &system
    ibrav=  2, celldm(1) ={10.2*supercell_size}, 
    nat=  {num_atoms}, 
    ntyp= 1,
    ecutwfc ={ecutwfc},
    occupations={occupation},
    smearing={smear}
    degauss={e_temp},
    nosym=.true.
 /
 &electrons
    diagonalization='davidson'
    mixing_mode = 'plain'
    mixing_beta = {mixing_beta}
    conv_thr =  {conv_thr}

 /
 &ions
    ion_temperature   = 'not_controlled'
    ion_velocities='from_input'
/
ATOMIC_SPECIES
 Si  28.086  Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS alat
    {coordinates_array}
K_POINTS automatic
   1 1 1 0 0 0
ATOMIC_VELOCITIES
    {velocities_array}
"""


with open(args.output, 'w') as file:
    file.write(input_template)
print("Input file generated successfully!")
