# TDDFT Accuracy Benchmark

We tested the accuracy of our implementation of the TDDFT method by comparing the absorption spectrum of the Methane molecule (CH4) calculated using QuantumMASALA with the results obtained from the CE-TDDFT code [^1] (for Quantum ESPRESSO v7.2). The Methane molecule was placed in a box of size 30 Bohrs and the calculations were performed using PBE functional and SG15v2 set of ONCV pseudopotentials, with a plane-wave cutoff of 25 Ry. The C-H bond length was set to 1.114 Å. The Methane molecule was excited using a delta pulse at time t=0 and the time propagation was performed for 10,000 time steps of roughly 2.4 attoseconds. The absorption spectrum was calculated using the Fourier transform of the dipole moment, as illustrated in the `examples` directory.

[^1]: X. Qian et al. Phys. Rev. B 73, 035408 (2006)

Files in this directory:
- `ch4.scf.in`: Input file for the SCF calculation, in Quantum ESPRESSO v7.2.
- `ch4.tddft_z.in`: Input file for the TDDFT calculation, in CE-TDDFT code.
- `ch4_scf_tddft.py`: Python script to run the SCF and TDDFT calculations in Quantum MASALA.