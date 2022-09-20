from typing import Optional

import numpy as np

from .cryst import ReciprocalLattice
from .constants import TPI

ROUND_PREC = 5
GOOD_PRIMES = [2, 3, 5, 7]


def gen_grid_shape(recilat: ReciprocalLattice, ecut: float):
    omega = np.sqrt(4 * ecut)
    bi = np.linalg.norm(recilat.recvec, axis=0)
    ni = [int(n) for n in np.ceil(2 * omega / bi)]

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
    recilat: ReciprocalLattice
    grid_shape: tuple[int, int, int]
    ecut: float
    gridmask: np.ndarray

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

        xi = [np.fft.fftfreq(n, 1 / n).astype("i4") for n in self.grid_shape]
        g_cryst = np.array(np.meshgrid(*xi, indexing="ij")).reshape(3, -1)
        g_2 = self.recilat.norm2(g_cryst)

        self.gridmask = g_2 <= 2 * self.ecut
        icut = np.nonzero(self.gridmask)[0]
        if len(icut) < 2:
            raise ValueError("'ecut' value too small. "
                             f"Only {len(icut)} points within 'euct'={self.ecut}"
                             )

        self.numg = len(icut)
        self.cryst = np.array(g_cryst[:, icut], dtype="i4", order="C")
        self.norm2 = np.array(g_2[icut], dtype="f8")
        self.idxgrid = np.unravel_index(icut, self.grid_shape)

        self.realspc_cellvol = TPI ** 3 / self.recilat.cellvol
        self.realspc_dv = self.realspc_cellvol / np.prod(self.grid_shape)

    @property
    def cart(self):
        return self.recilat.cryst2cart(self.cryst)

    @property
    def tpiba(self):
        return self.recilat.cryst2tpiba(self.cryst)
