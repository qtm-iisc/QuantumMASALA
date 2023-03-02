#!/bin/bash

export OMP_NUM_THREADS=1
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"
for pow in $(seq 0 1 4); do
    nproc=$((2 ** pow))

    echo "Running QE in $nproc processes"
    mpirun -np $nproc pw.x -nk 1 -nb $nproc -in si5x5x5.scf.in > $nproc.qe.out
done

