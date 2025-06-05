#!/bin/bash
##first run the python file

nohup mpirun -np 40 python si_scf_supercell_random.py 3 > 02_scf.out

##If in_files directory is present, then remove it
if [ -d "in_files" ]; then
    rm -r in_files
fi

#create a new in_files directory
mkdir in_files

#Move all the .in files created by the python script to the in_files directory
mv *.in in_files

# Navigate to the directory containing the input files
cd in_files || exit

#create a file named si.out
touch si.out

# Loop through all si_config_x.in files
for infile in si_*.in; do
    # Extract the base name without extension
    base_name="${infile%.in}"
    
    # Run the command using nohup
    nohup mpirun -np 40 pw.x -i "$infile" >> "si.out"
done

