#!/bin/bash

# Set the number of processors
processors=(48)

supercell=8
calctype=${supercell}x${supercell}x${supercell}

infile="SiSupercell${calctype}-100.scf.in"
date=$(date +"%d%b%Y")
outdir="run_gspace_${calctype}_${date}"

# Create the output directory
mkdir -p $outdir



# Loop through the processors array
for p in "${processors[@]}"
do
    sleep 10
    echo "Running qtm with $p processors"
    /usr/bin/time -va -o $outdir/Si_qtm_nt_${p}.scf.out mpirun -n $p python ../../../src/qtm/qe/inp/pw.py $infile -nt $p > $outdir/Si_qtm_nt_${p}.scf.out
done

for p in "${processors[@]}"
do
    sleep 10
    echo "Running QE with $p processors"
    /usr/bin/time -va -o $outdir/Si_qe_noscalapack_nt_$p.scf.out mpirun -n $p $QE_bin/pw.x -nb 1 -ni 1 -nk 1 -nt $p -nd 1 -i $infile > $outdir/Si_qe_noscalapack_nt_$p.scf.out
done