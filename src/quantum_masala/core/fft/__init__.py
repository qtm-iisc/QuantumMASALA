"""FFT Module of QuantumMASALA

This module contains interfaces that perform Discrete Fourier Transform of
quantities periodic across lattice. As Plane-Wave methods require the Fourier
Space to be truncated (by a specified Kinetic Energy Cutoff), they are
implemented to return/accept quantities in the truncated space (as specified
by a `GSpace` instance).

Notes
-----
This module is designed to perform 3D Fast Fourier Transform via functions
provided by the following libraries: (ordered by preference)
- ``mkl_fft``
- ``pyfftw``
- ``scipy``
- ``numpy``

As most, if not all, Discrete Fourier Transform operations used here involves
truncation of the Fourier space (G-space), interface classes implemented
contains methods that directly transform components to and from the G-Space.

This truncation also allows us to minimize computation during transformation by
skipping trivial operations such as (I)FFT of zeros. Decomposing
3D FFTs to 3 sets of 1D FFTs and skipping FFTs that are trivial can speed up
operations.

As a result, we have two FFT Driver implementations named
``slabs`` and ``sticks`, the latter utilizing the optimization.
"""

from .interface import FFTGSpace, FFTGSpaceWfc

__all__ = ["FFTGSpace", "FFTGSpaceWfc"]
