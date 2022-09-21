"""Module describing the Reciprocal Lattice Vectors (G-vectors)
truncated by a kinetic energy cutoff.

`GSpace` represents the Fourier Space, truncated by a maximum
wavevector norm / Kinetic Energy cutoff. This implies all quantites
in Discrete Fourier Transform will have zero components set for all
wavevectors (G-vectors) outside the cutoff.
"""

from typing import Optional

import numpy as np

from .cryst import ReciprocalLattice
from .constants import TPI

ROUND_PREC = 5
GOOD_PRIMES = [2, 3, 5, 7]


def gen_grid_shape(recilat: ReciprocalLattice, ecut: float):
    """Generates the required grid shape from given cutoff energy

    The grid shape is computed via the Nyquist-Shannon Sampling Theorem
    with the numbers chosen to have only the numbers in `GOOD_PRIMES` as
    their prime factors. This is required for FFT routines to run optimally.
    """
    omega = np.sqrt(2 * ecut)  # ecut = 0.5 * |G_max|^2
    # Computing 2 \pi / |a_i| where a_i are the basis vectors of the bravais lattice
    ai = np.linalg.norm(recilat.recvec_inv, axis=1) * TPI
    tpibai = TPI / ai
    ni = [int(n) for n in np.ceil(2 * omega / tpibai)]

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


def check_grid_shape(grid_shape: tuple[int, int, int]):
    """Checks if `grid_shape` is of the right type and contains positive integers"""
    if len(grid_shape) != 3:
        raise ValueError(
            f"'grid_shape' must be a tuple of three integers. Got {grid_shape}"
        )
    for i in range(3):
        if not isinstance(grid_shape[i], int):
            raise ValueError(
                f"'grid_shape[{i}] must be an integer. Got {grid_shape[i]}'"
            )
        if grid_shape[i] <= 0:
            raise ValueError(
                f"'grid_shape[{i}] must be positive. Got{grid_shape[i]}"
            )


class GSpace:
    """Represents the Truncated Fourier Space in Reciprocal Lattice

    Generates and stores the list of G-vectors that are within
    Kinetic-Energy cutoff `ecut`.

    Attributes
    ----------
    recilat : ReciprocalLattice
        Reciprocal Lattice of the crystal
    ecut : float
        Kinetic Energy Cutoff; Must be positive and large enough to contain atleast
        2 G-vectors within it
    grid_shape : tuple[int, int, int]
        Shape of the FFT Grid
    grid_mask : np.ndarray
        3D Boolean array; containing g-vectors
    numg : int
        Number of G-vectors within `ecut`
    cryst : np.ndarray
        G-vectors in crystal coords
    norm2 : np.ndarray
        Norm squared of the G-vectors in atomic units
    idxgrid : tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices of the FFT grid corresponding to the G-vectors

    reallat_cellvol : float
        Volume of the Unit cell in the bravais lattice corresponding to `recilat`
    reallat_dv : float
        Volume differential of integrals of the unit cell of bravais lattice


    """
    recilat: ReciprocalLattice
    ecut: float
    grid_shape: tuple[int, int, int]
    grid_mask: np.ndarray
    numg: int
    cryst: np.ndarray
    norm2: np.ndarray
    idxgrid: tuple[np.ndarray, np.ndarray, np.ndarray]
    reallat_cellvol: float
    reallat_dv: float
    
    __slots__ = ['recilat', 'ecut', 'grid_shape', 'grid_mask',
                 'numg', 'cryst', 'norm2', 'idxgrid',
                 'reallat_cellvol', 'reallat_dv'
                 ]

    def __init__(
        self, recilat: ReciprocalLattice, ecut: float,
            grid_shape: Optional[tuple[int, int, int]] = None,
    ):
        self.recilat = recilat

        if ecut <= 0:
            raise ValueError(f"'ecut' must be positive. Got {ecut}")
        self.ecut = ecut

        if grid_shape is None:
            grid_shape = gen_grid_shape(self.recilat, self.ecut)
        check_grid_shape(grid_shape)
        self.grid_shape = grid_shape
        
        # Generating all points in FFT grid
        xi = [np.fft.fftfreq(n, 1 / n).astype("i4") for n in self.grid_shape]
        g_cryst = np.array(np.meshgrid(*xi, indexing="ij")).reshape(3, -1)
        
        # Computing KE and truncating the list of G-vectors
        g_2 = self.recilat.norm2(g_cryst)
        self.grid_mask = g_2 <= 2 * self.ecut
        icut = np.nonzero(self.grid_mask)[0]
        if len(icut) < 2:
            raise ValueError("'ecut' value too small. "
                             f"Only {len(icut)} points within 'euct'={self.ecut}"
                             )

        # Storing data of all G-vectors within cutoff
        self.numg = len(icut)
        self.cryst = np.array(g_cryst[:, icut], dtype="i4", order="C")
        self.norm2 = np.array(g_2[icut], dtype="f8")
        self.idxgrid = np.unravel_index(icut, self.grid_shape)
        
        # Additional quantities required for integrating quantities across a unit cell of bravais lattice
        self.reallat_cellvol = TPI ** 3 / self.recilat.cellvol
        self.reallat_dv = self.reallat_cellvol / np.prod(self.grid_shape)

    @property
    def cart(self):
        """Returns the cartesian coords of the G-vectors in Atomic units"""
        return self.recilat.cryst2cart(self.cryst)

    @property
    def tpiba(self):
        """Returns the cartesian coords of the G-vectors in `tpiba` units"""
        return self.recilat.cryst2tpiba(self.cryst)
