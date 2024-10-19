#!/bin/bash

pw="mpirun -np 28 $QE_BIN/pw.x"
tddft="mpirun -np 28 $CE_TDDFT/tddft.x"

molecule=ch4

$pw <$molecule.scf.in > $molecule.scf.out

for edir in z
do
  $tddft <$molecule.tddft_$edir.in > $molecule.tddft_$edir.out
  grep ^DIP $molecule.tddft_$edir.out >dip_$edir.dat
done

