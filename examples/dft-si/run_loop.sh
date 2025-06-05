#!/bin/bash
for supercell_size in 6; do
  for processes in 8 16 32 64; do
    #/usr/bin/time -v mpirun -n $processes python memopt_si_scf_supercell.py $supercell_size > out_supercell_${supercell_size}_proc_${processes}.log 2>&17
    /usr/bin/time -v mpirun -n $processes ~/codes/q-e-qe-7.2/bin/pw.x -i SiSupercell6x6x6-100.scf.in > pwscf_out_supercell_${supercell_size}_proc_${processes}.log 2>&1
    /usr/bin/time -v mpirun -n $processes python /home/agrimsharma/QuantumMASALA_April2025_memory_optimization_review/QuantumMASALA/src/qtm/interfaces/qe/pw.py SiSupercell6x6x6-100.scf.in > pwqtm_out_supercell_${supercell_size}_proc_${processes}.log 2>&1
  done
done