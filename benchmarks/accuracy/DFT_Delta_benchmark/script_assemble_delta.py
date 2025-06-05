import glob
import os


bohr_to_ang = 0.529177249
ryd_to_ev = 13.605698066

print("# QuantumMASALA Delta Calculation 2023\n")

for filepath in glob.glob("./*/*.eosout"):
    # print("file:",filepath)
    # temp_dict = {}
    # with open(filepath, 'r') as f:
    #     for line in f.readlines():
    directory_name =  os.path.basename(os.path.dirname(filepath))
    atomic_number = int(directory_name.split('-')[0])
    elem_name = directory_name.split('-')[1]
    # print(directory_name, elem_name, atomic_number)

    with open(filepath, 'r') as f:    
        numbers = f.readlines()[2]
        print(f"{elem_name}     ",numbers.strip())