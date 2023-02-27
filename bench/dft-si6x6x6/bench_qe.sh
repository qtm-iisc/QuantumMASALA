#!/bin/bash

export OMP_NUM_THREADS=1
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"
for pow in $(seq 1 1 5); do
    nproc=$((2 ** pow))

    echo "Running QE in $nproc processes"
    mpirun -np $nproc pw.x -nk $nproc -in si6x6x6.scf.in > $nproc-$OMP_NUM_THREADS.qe.out
done
