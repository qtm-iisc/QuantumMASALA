#!/bin/bash

export OMP_NUM_THREADS=2
echo "Setting OMP_NUM_THREADS to $OMP_NUM_THREADS"
for pow in $(seq 1 1 5); do
    nproc=$((2 ** pow))

    echo "Running QTM in $nproc processes"
    mpirun -np $nproc python -u si5x5x5_scf.py > $nproc.py.out
done

echo "Running QTM in GPU"
python si5x5x5_scf.py --use-gpu > gpu.py.out
