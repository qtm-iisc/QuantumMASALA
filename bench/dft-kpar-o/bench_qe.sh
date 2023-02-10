#!/bin/bash

export OMP_NUM_THREADS=1
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"
for pow in $(seq 2 1 6); do
    nproc=$((2 ** pow))

    echo "Running QE in $nproc processes"
    mpirun -np $nproc pw.x -nk $nproc -in o.scf.in > $nproc.qe.out
done
