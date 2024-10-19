# I have a directory, on another computer, which contains runs for scf-delta - a benchmark for accuracy. The directory structure is as follows: rootdir/<atomic_number_integer>-<element_name>/<element-name>-<percent_of_actualvolume>
    
#     I have a directory, on another computer, which contains runs for scf-delta - a benchmark for accuracy. The directory structure is as follows: rootdir/<atomic_number_integer>-<element_name>/<element-name>-<percent_of_actualvolume>. There are several elements and several volumes for each element. within each volume's directory, there is a file called calc.log. Extract the folllowing lines from this file: ```     number of atoms/cell      = 4
#      unit-cell volume          = 321.1510696465894
# !    total energy              = -45.61418705740809```

# Nice! Now, as shown in the following snippet for similar code, convert the energy to eV and unit cell to Angstroms, and assembe volume and create a file in the element's directory, names <element-name>.txt and write `<volume per atom in angstrom> <energy in eV>` in that file

import os

# Constants
bohr_to_ang = 0.529177249
ryd_to_ev = 13.605698066

rootdir = './'

def text_from_log():
    for dirpath, dirnames, filenames in os.walk(rootdir):
        print(dirpath)
        # dirnames = sorted(dirnames, key=lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'))
        # for dirname in dirnames:
        for filename in filenames:
            if filename == 'calc.log':
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if 'number of atoms/cell' in line or 'unit-cell volume' in line or '!    total energy' in line:
                            print(line.strip())
                            
                            
                            if 'number of atoms/cell' in line:
                                n_atoms = int(line.strip().split()[-1])
                                # break

                            # Now that element name and atomic number is captured, we will read all the volumes and energies
                            # and write them to txt file.
                            # ASSUMPTION: volumes are in order
                            if 'unit-cell volume' in line:
                                volume = float(line.strip().split()[3])
                                volume *= (bohr_to_ang ** 3)
                                # break

                            if '!    total energy' in line:
                                volume /= n_atoms

                                energy = float(line.strip().split()[4])
                                energy *= ryd_to_ev
                                energy /= n_atoms
                                # Now write to file
                                print(dirpath)
                                elem_id = dirpath.split('/')[-2]  # Read element ID from dirpath
                                elem_name = elem_id.split('-')[-1]
                                print(f"{rootdir}/{elem_id}/{elem_name}.txt")
                                print(f"{elem_id}: {volume} {energy}")
                                with open(f"{rootdir}/{elem_id}/{elem_name}.txt","at") as dir_elem:
                                    dir_elem.write(f"{volume} {energy}\n")
                                #     print(f"{elem_id}: {volume} {energy}")
                                    
def rearrange_txt_lines_volume():
    # Iterate over all subdirectories in the root directory
    for dirpath, dirnames, filenames in os.walk(rootdir):
        # Iterate over all files in the current subdirectory
        for filename in filenames:
            if filename.endswith(".txt"):
                filepath = os.path.join(dirpath, filename)

                # Read all lines from the file
                with open(filepath, "r") as file:
                    lines = file.readlines()

                # Check if there are exactly 7 lines in the file
                if len(lines) == 7:
                    # Rearrange the lines
                    lines = lines[4:7] + lines[:4]

                    # Write the lines back to the file
                    with open(filepath, "w") as file:
                        file.writelines(lines)                               

text_from_log()
rearrange_txt_lines_volume()