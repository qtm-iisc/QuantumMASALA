from __future__ import annotations
from typing import Optional, Type
from qtm.config import NDArray
__all__ = ['GSpace']

import numpy as np

from qtm.lattice import ReciLattice

from .gspc_base import GSpaceBase
from .fft.utils import check_shape


ROUND_PREC: int = 6
"""Rounding precision of float used when sorting.
"""
GOOD_PRIMES: list[int] = [2, 3, 5, 7]
"""List of primes used for finding optimal FFT grid lengths.
"""


def gen_grid_shape(recilat, ecut) -> tuple[int, ...]:
    """Generates optimal FFT grid shape from given cutoff energy

    Parameters
    ----------
    recilat : ReciLattice
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
    if not isinstance(recilat, ReciLattice):
        raise ValueError("'recilat' must be a 'ReciLattice' instance. "
                         f"got {type(recilat)}")
    if ecut <= 0:
        raise ValueError(f"'ecut' must be a positive number. got {ecut}")

    omega = np.sqrt(2 * ecut)  # ecut = 0.5 * |G_max|^2
    # Computing 2 \pi / |a_i| where a_i are the basis vectors of the bravais lattice
    b1, b2, b3 = recilat.axes_cart
    r1 = recilat.cellvol / np.linalg.norm(np.cross(b2, b3))
    r2 = recilat.cellvol / np.linalg.norm(np.cross(b1, b3))
    r3 = recilat.cellvol / np.linalg.norm(np.cross(b1, b2))
    ni = [2*int(np.floor(omega/r)) + 1 for r in [r1, r2, r3]]
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


class GSpace(GSpaceBase):

    def __init__(self, recilat: ReciLattice, ecut: float,
                 grid_shape: Optional[tuple[int, int, int]] = None):
        if not isinstance(recilat, ReciLattice):
            raise ValueError("'recilat' must be an instance of 'ReciprocalLatvec'. "
                             f"got {type(recilat)}")

        if ecut <= 0:
            raise ValueError(f"'ecut' must be positive. Got: {ecut}")
        self.ecut: float = ecut
        """Kinetic Energy Cutoff
        """

        if grid_shape is None:
            grid_shape = gen_grid_shape(recilat, self.ecut)
        else:
            check_shape(grid_shape)
        """Shape of the 3D FFT Grid containing G-vectors
        """

        # Generating all points in FFT grid
        xi = [np.arange(-(n//2), (n + 1)//2) for n in grid_shape]
        g_cryst = np.array(np.meshgrid(*xi, indexing="ij"),
                           like=recilat.primvec).reshape(3, -1)

        # Computing KE and truncating the list of G-vectors
        g_2 = recilat.norm2(g_cryst)
        ke_max = 0.5 * np.amax(g_2)
        if ke_max <= self.ecut:
            raise ValueError("'ecut' value too large. Largest KE in PW Basis "
                             f"is {ke_max} Hart. Given 'ecut'={self.ecut} Hart"
                             )

        grid_mask = g_2 <= 2 * self.ecut
        icut = np.nonzero(grid_mask)[0]
        if len(icut) < 2:
            raise ValueError("'ecut' value too small. "
                             f"Only {len(icut)} points within "
                             f"'euct'={self.ecut} Hart")

        # Ordering G-vectors in ascending order of lengths
        tpiba = recilat.tpiba
        isort = np.argsort(np.around(g_2[icut] / tpiba, ROUND_PREC),
                           kind='stable')
        icut = icut[isort]

        g_cryst = g_cryst[(slice(None), icut)]
        super().__init__(recilat, grid_shape, g_cryst)

    def __eq__(self, other):
        # Check if both reference the same object
        if other is self:
            return True

        # Check if they are also a 'GSpace' instance
        if not isinstance(other, GSpace):
            return False
        # Check if they represent the same lattice
        if other.recilat != self.recilat:
            return False
        # Check if they have same cutoff
        if other.ecut != self.ecut:
            return False
        # By here both should contain the same list of G-vectors.
        # But, the FFT grid shape can be different, if user specified.
        return other.grid_shape == self.grid_shape
