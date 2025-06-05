#!/bin/bash

# Set the number of processors
processors=(28 14 8 4 2 1)

calctype="6x6x6"

infile="SiSupercell${calctype}-100.scf.in"

outdir="."

# Create the output directory
mkdir -p $outdir



for p in "${processors[@]}"
do
    sleep 10 
    echo "Running qtm with $p processors"
    time mpirun -n $p python ../../../src/qtm/qe/inp/pw.py $infile -nb $p > $outdir/Si_qtm_nb_$p.scf.out
done

for p in "${processors[@]}"
do
    sleep 10
    echo "Running QE with $p processors"
    time mpirun -n $p $QE_bin/pw.x -nb $p -ni 1 -nk 1 -nt 1 -nd 1 -i $infile > $outdir/Si_qe_lapack_nb_$p.scf.out
done