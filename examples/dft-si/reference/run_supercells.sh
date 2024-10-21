#!/bin/bash

for size in 1 2 3 4 5 6; do
    mpirun -np $((2 * size * size)) python si_scf_supercell.py $size > si_scf_supercell_$size.out
done