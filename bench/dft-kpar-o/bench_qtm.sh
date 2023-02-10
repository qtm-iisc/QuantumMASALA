#!/bin/bash

export OMP_NUM_THREADS=1
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"
for pow in $(seq 2 1 6); do
    nproc=$((2 ** pow))

    echo "Running QTM in $nproc processes"
    mpirun -np $nproc python -u o_scf.py > $nproc.py.out
done
