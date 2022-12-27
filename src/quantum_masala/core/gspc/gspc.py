"""Truncated Reciprocal Space (G-Space) Module

``GSpace`` represents the Fourier Space that is truncated by a maximum
wavevector norm / Kinetic Energy cutoff. This implies all Discrete
Fourier Transform Components corresponsing to wavevectors outside
the cutoff will be set to zero.
"""

__all__ = ["GSpace"]

import numpy as np

from quantum_masala import pw_counter
from quantum_masala.core import Crystal, ReciprocalLattice, fft
from quantum_masala.constants import TPI
from .gspc_symm import SymmMod


GOOD_PRIMES: list[int] = [2, 3, 5]
"""List of primes used for finding optimal FFT grid lengths.
"""

ROUND_PREC: int = 6
"""Rounding precision of float used when sorting.
"""


def _gen_grid_shape(recilat, ecut) -> tuple[int, ...]:
    """Generates the optimal FFT grid shape from given cutoff energy

    Parameters
    ----------
    recilat : ReciprocalLattice
        Reciprocal Lattice
    ecut : float
        Kinetic Energy cutoff

    Returns
    -------
    grid_shape : tuple[int, int, int]
        FFT Grid Shape

    Notes
    -----
    The grid shape is computed via the Nyquist-Shannon Sampling Theorem
    with the numbers chosen to have only the numbers in ``GOOD_PRIMES`` as
    their prime factors. This is required as FFT Libraries perform optimally
    for sizes with only good prime factors (2, 3, 5, 7).
    """
    omega = np.sqrt(2 * ecut)  # ecut = 0.5 * |G_max|^2
    # Computing 2 \pi / |a_i| where a_i are the basis vectors of the bravais lattice
    ai = np.linalg.norm(recilat.recvec_inv, axis=1) * TPI
    tpibai = TPI / ai
    ni = [int(n) for n in np.floor(2 * omega / tpibai)]

    # Finding 'good' values of `grid_shape`
    grid_shape = []
    for x in ni:
        is_good = False
        while not is_good:
            n = x
            for prime in GOOD_PRIMES:
                while n % prime == 0:
                    n = n / prime
            if n == 1:
                is_good = True
            else:
                x += 1
        grid_shape.append(x)
    return tuple(grid_shape)


class GSpace:
    """Represents the Fourier Space in Reciprocal Lattice truncated by Kinetic
    Energy cutoff

    Container for G-vectors within cutoff and the corresponding FFT Module for
    performing FFT between real space and the truncated fourier space.

    Parameters
    ----------
    crystal : Crystal
        Represents the unit cell of Crystal
    ecut : float
        Kinetic Energy Cutoff; Must be positive and large enough to contain atleast
        2 G-vectors within it
    grid_shape : tuple[int, int, int], optional
        Shape of the 3D Grid
    """

    __slots__ = ['recilat', 'ecut', 'grid_shape', 'grid_mask',
                 'numg', 'cryst', 'norm2', 'idxgrid',
                 'reallat_cellvol', 'reallat_dv', 'fft_mod', 'symm_mod',
                 ]

    def __init__(self, crystal: Crystal, ecut: float,
                 grid_shape: tuple[int, int, int] = None):
        pw_counter.start_clock('gspc:init')
        self.recilat: ReciprocalLattice = crystal.recilat
        """Reciprocal Latiice of the crystal.
        """

        if ecut <= 0:
            raise ValueError(f"'ecut' must be positive. Got: {ecut}")
        self.ecut: float = ecut
        """Kinetic Energy Cutoff
        """

        if grid_shape is None:
            grid_shape = _gen_grid_shape(self.recilat, self.ecut)
        self.grid_shape: tuple[int, int, int] = grid_shape
        """Shape of the 3D FFT Grid containing G-vectors
        """

        # Generating all points in FFT grid
        xi = [np.fft.fftfreq(n, 1 / n).astype("i4") for n in self.grid_shape]
        g_cryst = np.array(np.meshgrid(*xi, indexing="ij")).reshape(3, -1)

        # Computing KE and truncating the list of G-vectors
        g_2 = self.recilat.norm2(g_cryst)
        self.grid_mask: np.ndarray = g_2 <= 2 * self.ecut
        """(``self.grid_shape``, ``'bool'``) Boolean Mask
        indicating G-vectors within the 3D FFT Grid
        """

        icut = np.nonzero(self.grid_mask)[0]
        if len(icut) < 2:
            raise ValueError("'ecut' value too small. "
                             f"Only {len(icut)} points within 'euct'={self.ecut}"
                             )

        # Storing data of all G-vectors within cutoff
        idxsort = np.lexsort((g_cryst[2, icut], g_cryst[1, icut], g_cryst[0, icut],
                              np.around(g_2[icut], ROUND_PREC)
                              ))
        icut = icut[idxsort]

        self.numg: int = len(icut)
        """Number of G-vectors (`int`)
        """
        self.cryst: np.ndarray = np.array(g_cryst[:, icut], dtype="i4", order="C")
        """(``(3, self.numg)``, ``'i4'``, ``'C'``)
        Crystal Coordinates of G-vectors
        """
        self.norm2: np.ndarray = np.array(g_2[icut], dtype="f8")
        """(``(self.numg, )``, ``'f8'``) Norm Squared of
        G-vectors
        """
        self.idxgrid: tuple[np.ndarray, ...] = np.unravel_index(icut, self.grid_shape)
        """Position of G-vectors in the 3D FFT Grid
        """

        # Additional quantities required for integrating quantities across the
        # unit cell of bravais lattice
        self.reallat_cellvol: float = TPI ** 3 / self.recilat.cellvol
        """Volume of unit cell of the real lattice corresponding to
        ``self.recilat``
        """
        self.reallat_dv: float = self.reallat_cellvol / np.prod(self.grid_shape)
        """Volumme differential 'dv' for integrals across the real 
        lattice corresponding to ``self.recilat``
        """

        self.fft_mod: fft.base.FFTModule = fft.get_fft_module()(self.grid_shape, self.idxgrid)
        """FFT Module to perform Fourier Transform between 'r'eal and 'g'-space
        """

        self.symm_mod = SymmMod(crystal, self)
        """Symmetrization Module to ensure values have same symmetry as crystal
        """
        pw_counter.stop_clock('gspc:init')

    @property
    def cart(self) -> np.ndarray:
        """(``(3, self.numg)``, ``'f8'``) Cartesian coords of
        the G-vectors in Atomic units
        """
        return self.recilat.cryst2cart(self.cryst)

    @property
    def tpiba(self) -> np.ndarray:
        """(``(3, self.numg)``, ``'f8'``) Cartesian coords of
        the G-vectors in 'tpiba' units
        """
        return self.recilat.cryst2tpiba(self.cryst)
