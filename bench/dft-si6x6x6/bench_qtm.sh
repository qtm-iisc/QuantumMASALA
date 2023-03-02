#!/bin/bash

export OMP_NUM_THREADS=1
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"
for pow in $(seq 5 -1 0); do
    nproc=$((2 ** pow))

    echo "Running QTM in $nproc processes"
    mpirun -np $nproc python -u si6x6x6_scf.py -nk 1 > $nproc-$OMP_NUM_THREADS.py.out
done
