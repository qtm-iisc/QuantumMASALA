# GW Accuracy Benchmark

For testing the accuracy of GW calculations, we compared quasiparticle energy calculation results from BerkeleyGW and from Quantum MASALA for Static COHSEX, and Hybertsen-Louie Plasmon Pole with Static Remainder correction in table (1) in the paper. For the mean field calculation (using DFT) on a 6 × 6 × 6 Monkhorst-Pack k-grid. The wavefunction was expanded in plane waves with energy upto 25 Ry. A dielectric cut off of 25 Ry was used. 274 empty bands were included in CH summation. The calculated quasiparticle energies obtained using Quantum MASALA were found to match the results obtained using BerkeleyGW to within 100 μeV.

Files in this directory:

- `si.scf.in`: Input file for the self-consistent field calculation, in Quantum ESPRESSO.
- `si_q.scf.in`: Input file for the self-consistent field calculation, in Quantum ESPRESSO, with a shifted k-point grid.
- `si.pw2bgw.in`: Input file for the conversion of Quantum ESPRESSO output to BerkeleyGW input.
- `si_q.pw2bgw.in`: Input file for the conversion of Quantum ESPRESSO output to BerkeleyGW input, with a shifted k-point grid.
- `epsilon.inp`: Input file for the epsilon calculation, in BerkeleyGW.
- `sigma.inp`: Input file for the sigma calculation, in BerkeleyGW.
    - Please modify the input variables in this file to run a Static COHSEX calculation, instead of a Hybertsen-Louie Plasmon Pole calculation, by changing `frequency_dependence` to `0`.

To understand the workflow of the benchmark GW calculations, please refer to BerkeleyGW examples and documentation. The files `epsilon_script.py` and `sigma.py` are the Quantum MASALA alternatives for BerkeleyGW's `epsilon.cplx.x` and `sigma.cplx.x`, respectively.