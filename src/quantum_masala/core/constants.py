"""Units and Constants used across QuantumMASALA.

Values of Physical Constants from NIST (2018)
https://physics.nist.gov/cuu/Constants/index.html
"""
from numpy import pi, sqrt

PI = pi
TPI = 2 * PI
FPI = 4 * PI
TPIJ = TPI * 1j

e_SI = 1.602176634e-19  # C
k_SI = 1.380649e-23  # JK^-1
eV_SI = 1.602176634e-19  # J
bohr_SI = 5.29177210903e-11  # m
angstrom = 1e-10  # m

E_hart_SI = 4.3597447222071e-18  # J
E_ryd_SI = E_hart_SI / 2  # J
q_hart_SI = e_SI  # C
q_ryd_SI = e_SI / sqrt(2)  # C


# Converting to Hartree Atomic Units

hart = 1
q_hart = 1
bohr = 1

e = e_SI / q_hart_SI
k = k_SI / E_hart_SI
eV = eV_SI / E_hart_SI
angstrom = angstrom / bohr_SI

E_ryd = 1.0 / 2
q_ryd = 1 / sqrt(2)
