#!/bin/bash

# Set the number of processors
# processors=(28)
processors=(64 56 32 16 14 8 4 2 1)

infile="Fe-100.scf.in"
outdir="kpoints_results"

# Create the output directory
mkdir -p $outdir

# Loop through the processors array

for p in "${processors[@]}"
do
    echo "Running with $p processors"
    echo "QTM"
    sleep 5
    time mpirun -n $p python ../../../src/qtm/qe/inp/pw.py Fe-100.scf.in -nk $p > $outdir/Fe_qtm_nk_$p.scf.out
done


for p in "${processors[@]}"
do
    echo "Running with $p processors"
    echo "QE"
    sleep 5
    time mpirun -n $p $QE_bin/pw.x -nk $p -nb 1 -nt 1 -nd 1 -i $infile > $outdir/Fe_qe_nk_$p.scf.out
done
