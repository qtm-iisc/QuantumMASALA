# Performance Benchmarks for QuantumMASALA

This directory contains performance benchmarks for QuantumMASALA. We measure the performance of QuantumMASALA by running the SCF calculations under various parallelization settings and measuring the time taken for the calculations to complete. We measure our numbers against the performance of QuantumESPRESSO, which is a widely used electronic structure code.

This directory contains the performance benchmarks for the following parallelization settings:
1. Only k-point parallelization (Fe)
2. Only band parallelization    (Si 6x6x6 supercell)
3. Only G-space parallelization (Si 6x6x6 supercell)

Each benchmark can be run by using the provided bash script in each subdirectory, which (after an appropriate modification/specification of paths of QE executable, `$QE_bin`), runs the same SCF calculation for QuantumMASALA and QuantumESPRESSO.

The folder also contains the runs for 8x8x8 Silicon supercell, under G-space parallelization, for QuantumMASALA and QuantumESPRESSO.

