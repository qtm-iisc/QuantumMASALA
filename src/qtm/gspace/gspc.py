from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
__all__ = ['GSpace']

import numpy as np

from qtm.lattice import ReciLattice
from .base import GSpaceBase

from qtm.msg_format import value_mismatch_msg

ROUND_PREC: int = 6
"""Rounding precision of float used when sorting.
"""
GOOD_PRIMES: list[int] = [2, 3, 5]
"""List of primes used for finding optimal FFT grid lengths.
"""


def optimal_grid_shape(shape: Sequence[int]) -> tuple[int, ...]:
    out = []
    for ni in shape:
        is_good = False
        while not is_good:
            x = ni
            for prime in GOOD_PRIMES:
                while x % prime == 0:
                    x = x / prime
            if x == 1:
                is_good = True
            else:
                ni += 1
        out.append(ni)
    return tuple(out)


def minimal_grid_shape(recilat: ReciLattice, ecut: float) -> tuple[int, int, int]:
    """Computes the smallest FFT mesh grid that contains all G-vectors that are
    within given Kinetic Energy Cutoff.

    Parameters
    ----------
    recilat : ReciLattice
        Reciprocal Lattice of the System
    ecut : float
        Kinetic Energy cutoff

    Returns
    -------
    grid_shape : tuple[int, int, int]
        Dimensions of the FFT mesh grid that contains all G-vectoors of given
        'recilat' that are within cutoff 'ecut'

    """
    if not isinstance(recilat, ReciLattice):
        raise ValueError(f"'recilat' must be a '{ReciLattice}' instance. "
                         f"got {type(recilat)}")
    if ecut <= 0:
        raise ValueError("'ecut' must be a positive number. "
                         f"got {ecut}")
    # Computing the radius of the G-Sphere
    omega = np.sqrt(2 * ecut)  # ecut = 0.5 * |G_max|^2

    b1, b2, b3 = recilat.axes_cart
    # Computing the spacing between lattice planes perpendicular to each basis vector
    r1 = recilat.cellvol / np.linalg.norm(np.cross(b2, b3))
    r2 = recilat.cellvol / np.linalg.norm(np.cross(b1, b3))
    r3 = recilat.cellvol / np.linalg.norm(np.cross(b1, b2))
    return tuple(2*int(np.floor(omega/r)) + 1 for r in [r1, r2, r3])


def check_grid_shape(recilat: ReciLattice, ecut: float,
                     grid_shape: tuple[int, int, int]):
    min_grid_shape = minimal_grid_shape(recilat, ecut)
    assert isinstance(grid_shape, tuple)
    assert all(isinstance(ni, int) and ni > 0 for ni in grid_shape)
    for idim, ni in enumerate(min_grid_shape):
        if ni < min_grid_shape[idim]:
            raise ValueError(
                "'grid_shape' is too small to fit all the G-vectors within "
                f"KE cutoff {ecut} Hart. \n"
                f"Minimal FFT grid required is {min_grid_shape}. got {grid_shape}"
            )


class GSpace(GSpaceBase):

    def __init__(self, recilat: ReciLattice, ecut: float,
                 grid_shape: tuple[int, int, int] | None = None):
        if not isinstance(recilat, ReciLattice):
            raise ValueError(f"'recilat' must be a '{ReciLattice}'. "
                             f"got {type(recilat)}")

        if ecut <= 0:
            raise ValueError(f"'ecut' must be positive. Got: {ecut}")
        self.ecut: float = ecut
        """Kinetic Energy Cutoff
        """

        if grid_shape is None:
            grid_shape = optimal_grid_shape(minimal_grid_shape(recilat, self.ecut))
        else:
            grid_shape = tuple(grid_shape)
            if not all(isinstance(ni, int) and ni > 0 for ni in grid_shape):
                raise ValueError(value_mismatch_msg(
                    'grid_shape', grid_shape, 'a tuple of 3 positive integers')
                )
            check_grid_shape(recilat, ecut, grid_shape)

        # Generating all points in FFT grid
        xi = [np.arange(-(n//2), (n + 1)//2) for n in grid_shape]
        g_cryst = np.array(np.meshgrid(*xi, indexing="ij"),
                           like=recilat.primvec).reshape(3, -1)

        # Computing KE and truncating the list of G-vectors
        g_2 = recilat.norm2(g_cryst)
        ke_max = 0.5 * np.amax(g_2)
        if ke_max <= self.ecut:
            raise ValueError("'ecut' value too large for given 'grid_shape'. "
                             f"Largest KE for grid_shape = {grid_shape} "
                             f"is {ke_max} Hart. Given 'ecut'={self.ecut} Hart"
                             )

        grid_mask = g_2 <= 2 * self.ecut
        icut = np.nonzero(grid_mask)[0]
        if len(icut) < 2:
            raise ValueError("'ecut' value too small. "
                             f"Only {len(icut)} points within "
                             f"'ecut'={self.ecut} Hart")

        # Ordering G-vectors in ascending order of lengths
        tpiba = recilat.tpiba
        isort = np.argsort(np.around(g_2[icut] / tpiba, ROUND_PREC),
                           kind='stable')
        icut = icut[isort]

        g_cryst = g_cryst[(slice(None), icut)]
        super().__init__(recilat, grid_shape, g_cryst)
