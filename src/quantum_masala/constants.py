"""Units and Constants used across QuantumMASALA.

Notes
-----
Values of Physical Constants from NIST (2018)
https://physics.nist.gov/cuu/Constants/index.html
"""

from numpy import pi, sqrt

# Mathematical Constants
PI = pi
SQRT_PI = sqrt(pi)
TPI = 2 * PI
FPI = 4 * PI
TPIJ = TPI * 1j

# Constants from NIST
ELECTRON_SI = 1.602176634e-19     # Charge of electron, in C
BOLTZMANN_SI = 1.380649e-23       # Boltzmann Constant, in JK^-1
BOHR_SI = 5.29177210903e-11       # Bohr Radius,        in m
HARTREE_SI = 4.3597447222071e-18  # Hartree Energy,     in J

# Units of Charge: Hartree atomic unit and Rydberg atomic unit
ELECTRON_HART = 1.
ELECTRON_RYD = 1. / sqrt(2)

# Units of Energy; Electron Volt, Hartree and Rydberg
ELECTRONVOLT_SI = ELECTRON_SI
RYDBERG_SI = HARTREE_SI / 2
ELECTRONVOLT_HART = ELECTRONVOLT_SI / HARTREE_SI
ELECTRONVOLT_RYD = ELECTRONVOLT_SI / RYDBERG_SI
RYDBERG_HART = 1. / 2

# Units of Length; Bohr and Angstrom
ANGSTROM_SI = 1e-10
ANGSTROM_BOHR = ANGSTROM_SI / BOHR_SI

BOHR = 1.
ANGSTROM = ANGSTROM_BOHR
ELECTRONVOLT = ELECTRONVOLT_HART
ELECTRON = 1.
RYDBERG = RYDBERG_HART
