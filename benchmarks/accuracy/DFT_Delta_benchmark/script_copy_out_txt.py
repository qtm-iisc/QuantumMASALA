""" Python script to copy output values to txt files 

Data format:
```
This run is for atomic numbers from 11 to 20 (both inclusive).
Running QuantumMASALA
Directories to calculate:
['1-H', '2-He', '3-Li', '4-Be', '5-B', '6-C', '7-N', '8-O', '9-F', '10-Ne', '11-Na', '12-Mg', '13-Al', '14-Si', '15-P', '16-S', '17-Cl', '18-Ar', '19-K', '20-Ca', '21-Sc', '22-Ti', '23-V', '24-Cr', '25-Mn', '26-Fe', '27-Co', '28-Ni', '29-Cu', '30-Zn', '31-Ga', '32-Ge', '33-As', '34-Se', '35-Br', '36-Kr', '37-Rb', '38-Sr', '39-Y', '40-Zr', '41-Nb', '42-Mo', '44-Ru', '45-Rh', '46-Pd', '47-Ag', '48-Cd', '49-In', '50-Sn', '51-Sb', '52-Te', '53-I', '54-Xe', '55-Cs', '56-Ba', '72-Hf', '73-Ta', '74-W', '75-Re', '76-Os', '77-Ir', '78-Pt', '79-Au', '80-Hg', '81-Tl', '82-Pb', '83-Bi']
----------Calculating Element 11-Na----------
-----Input File: Na-94.scf.in-----
     unit-cell volume          = 705.4506979667058
!    total energy              = -255.59855027806515
-----Input File: Na-96.scf.in-----
     unit-cell volume          = 720.4602872851456
!    total energy              = -255.59899465924178
-----Input File: Na-98.scf.in-----
     unit-cell volume          = 735.4698766035858
!    total energy              = -255.59927441154193
-----Input File: Na-100.scf.in-----
...
```
"""

# Constants
bohr_to_ang = 0.529177249
ryd_to_ev = 13.605698066


with open("/home/agrimsharma/codes/benchmark_Dec2023_QuantumMASALA/QuantumMASALA/bench/qtm_dft_delta/SCFout_qtm_on_1_to_83/grepped_values.out",
          "r") as f:
    for line in f.readlines():
        if line.startswith("----------Calculating Element"):
            elem_id = str(line[30:35])
            while elem_id.endswith("-"):
                elem_id = str(elem_id[:-1])
            atomic_no = int(elem_id.split("-")[0])
            elem_name = elem_id.split("-")[1]

            # Read number of atoms from input file
            with open(f"/home/agrimsharma/codes/benchmark_Dec2023_QuantumMASALA/QuantumMASALA/bench/qtm_dft_delta/SCFout_qtm_on_1_to_83/{elem_id}/{elem_name}-94.scf.in","r") as dir_input:
                for line in dir_input.readlines():
                    if line.startswith("   nat"):
                        n_atoms = int(line.strip().split()[-1])
            
            print(line, elem_id, atomic_no, elem_name, n_atoms)

        # Now that element name and atomic number is captured, we will read all the volumes and energies
        # and write them to txt file.
        # ASSUMPTION: volumes are in order
        if line.startswith("     unit-cell volume"):
            volume = float(line.strip().split()[-1])
            volume *= (bohr_to_ang ** 3)
            volume /= n_atoms
        if line.startswith("!    total energy"):
            energy = float(line.strip().split()[-1])
            energy *= ryd_to_ev
            energy /= n_atoms
            # Now write to file
            with open(f"/home/agrimsharma/codes/benchmark_Dec2023_QuantumMASALA/QuantumMASALA/bench/qtm_dft_delta/SCFout_qtm_on_1_to_83/{elem_id}/{elem_name}.txt","at") as dir_elem:
                dir_elem.write(f"{volume} {energy}\n")
                print(f"{elem_id}: {volume} {energy}")




