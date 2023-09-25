"""Units and Constants used across QuantumMASALA.

Values of Physical Constants taken from NIST (2018):
https://physics.nist.gov/cuu/Constants/index.html

Internally, QuantumMASALA uses Hartree Atomic Units. Any physical values
imported from here that do not have unit suffix (``_SI``, ``_RYD``,
``_HART``, etc.) are assumed to be in Hartree Units.

Notes
-----
Sphinx fails to parse docstrings for module variables. So docstrings are not
rendered. Refer: https://github.com/sphinx-doc/sphinx/issues/1063
"""

from numpy import pi, sqrt

EPS = 1E-5

PI = pi  #: Pi
SQRT_PI = sqrt(pi)  #: Square root of Pi
TPI = 2 * PI
FPI = 4 * PI
TPIJ = TPI * 1j

# Constants from NIST
ELECTRON_SI = 1.602176634e-19     #: Charge of electron in C
BOLTZMANN_SI = 1.380649e-23       #: Boltzmann Constant in JK^-1
BOHR_SI = 5.29177210903e-11       #: Bohr Radius in m
HARTREE_SI = 4.3597447222071e-18  #: Hartree Energy in J

# Units of Charge: Hartree atomic unit and Rydberg atomic unit
ELECTRON_HART = 1.                #: Charge of electron in Hartree Atomic Units
ELECTRON_RYD = sqrt(2)       #: Charge of electron in Rydberg Atomic Units

# Units of Energy: Electronvolt, Hartree and Rydberg
ELECTRONVOLT_SI = ELECTRON_SI     #: Electronvolt in J
RYDBERG_SI = HARTREE_SI / 2       #: Rydberg in J
ELECTRONVOLT_HART = ELECTRONVOLT_SI / HARTREE_SI  #: Electronvolt in Hartree Atomic Units
ELECTRONVOLT_RYD = ELECTRONVOLT_SI / RYDBERG_SI   #: Electronvolt in Rydberg Atomic Units
RYDBERG_HART = 1. / 2             #: Rydberg in Hartree Atomic Units

# Units of Length: Bohr and Angstrom
ANGSTROM_SI = 1e-10               #: Angstrom in m
ANGSTROM_BOHR = ANGSTROM_SI / BOHR_SI    #: Angstrom in Bohr

# Units used in QuantumMASALA (Hartree Atomic Units)
BOHR = 1.
ANGSTROM = ANGSTROM_BOHR
ELECTRONVOLT = ELECTRONVOLT_HART
ELECTRON = 1.
HARTREE = 1.
RYDBERG = RYDBERG_HART


