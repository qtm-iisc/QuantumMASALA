#!/bin/bash

mkdir -p results_cupy
for i in 1 2 3 4 5 6 
do
/usr/bin/time -v python si_scf_cupy.py $i > results_cupy/cupy_$i.out
done