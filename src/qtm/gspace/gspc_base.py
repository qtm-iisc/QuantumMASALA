# from __future__ import annotations
from typing import Optional, Union, Sequence
from qtm.config import NDArray
__all__ = ['GSpaceBase', ]

import numpy as np

from qtm.lattice import ReciLattice

from .fft import FFT3DFull
from .fft.utils import check_g_cryst, cryst2idxgrid

from qtm.constants import TPI


ROUND_PREC: int = 6
"""Rounding precision of float used when sorting.
"""


class GSpaceBase:

    FFT3D = FFT3DFull
    _normalise_idft = True

    def __init__(self, recilat: ReciLattice, grid_shape: tuple[int, int, int],
                 g_cryst: NDArray):
        self.recilat: ReciLattice = recilat
        """Reciprocal Lattice"""

        check_g_cryst(grid_shape, g_cryst)
        self.grid_shape: tuple[int, int, int] = grid_shape
        """3D FFT Mesh dimensions"""

        idxgrid = cryst2idxgrid(self.grid_shape, g_cryst)
        idxsort = np.argsort(idxgrid)

        self.g_cryst: NDArray = g_cryst[(slice(None), idxsort)].copy()
        """(``(size, 3)``, ``'i8'``) Crystal coordinates of the G vectors/
        FFT frequencies"""
        self.idxgrid = idxgrid[idxsort]
        """(``(size, )``, ``'i8'``) (Flat) Indices mapping G vectors in `g_cryst`
        to the position in the (flattened) 3D FFT meshgrid"""
        self.idxsort = np.argsort(idxsort)
        """(``(size, )``, ``'i8'``) Indices that reorders the G vectors
        in `g_cryst` to match the input"""

        self.size_r: int = int(np.prod(self.grid_shape))
        """Size of the Real-Space i.e Number of points in the FFT Grid"""
        self.size_g: int = self.g_cryst.shape[1]
        """Size of the G-Space i.e. Number of G-vectors"""
        self.has_g0: bool = np.all(self.g_cryst[:, 0] == 0)
        """If True, contains G=0 vector. Will always be the first element
        i.e `g_cryst[:, 0]`"""

        self.reallat_cellvol: float = TPI**3 / self.recilat.cellvol
        """Volume of a unit cell of the real-space lattice"""
        self.reallat_dv: float = self.reallat_cellvol / np.prod(self.grid_shape)
        """Differential volume used when evaluating integrals across a unit-cell
        of the real-space lattice"""
        self._fft = self.FFT3D(
            self.grid_shape, self.idxgrid, normalise_idft=self._normalise_idft
        )
        """FFT Module"""

    @property
    def g_cart(self) -> NDArray:
        """(``(3, size)``, ``'f8'``) Cartesian coordindates of G vectors."""
        return self.recilat.cryst2cart(self.g_cryst)

    @property
    def g_tpiba(self) -> NDArray:
        r"""(``(3, size)``, ``'f8'``) Cartesian coordinates of G vectors in
        units of `tpiba` (:math:`\frac{2\pi}{a}`)."""
        return self.recilat.cryst2tpiba(self.g_cryst)

    @property
    def g_norm2(self) -> NDArray:
        """(``(size, )``, ``'f8'``) Norm squared of G vectors."""
        return self.recilat.norm2(self.g_cryst, 'cryst')

    @property
    def g_norm(self) -> NDArray:
        """(``(size, )``, ``'f8'``) Norm of G vectors."""
        return np.sqrt(self.g_norm2)

    def __eq__(self, other) -> bool:
        return other is self

    def create_buffer(self, shape: Union[int, Sequence[int]]) -> NDArray:
        """Returns an empty array of given shape.

        Parameters
        ----------
        shape : tuple[int, ...]
            shape of the array to create

        Returns
        -------
        NDArray
            Empty buffer of given shape
        """
        return self._fft.create_buffer(shape)

    def check_buffer(self, arr: NDArray) -> None:
        """Checks if input buffer is of right type, contiguous, etc.

        Parameters
        ----------
        arr : NDArray
            Input buffer to be validated.

        Raises
        -------
        ValueError
            Raised if input buffer fails any checks
        """
        self._fft.check_buffer(arr)

    def create_buffer_r(self, shape: Union[int, Sequence[int]]) -> NDArray:
        """Returns an empty buffer for storing real-space field of given shape.

        Parameters
        ----------
        shape : tuple[int, ...]
            shape of the list of real-space buffers

        Returns
        -------
        NDArray
            Empty buffer of shape ``(*shape, *grid_shape)``
        """
        if shape == ():
            return self.create_buffer((self.size_r, ))
        if isinstance(shape, int):
            shape = (shape, )
        return self.create_buffer((*shape, self.size_r))

    def create_buffer_g(self, shape: Union[int, Sequence[int]]) -> NDArray:
        """Returns an empty buffer for storing g-space field of given shape.

        Parameters
        ----------
        shape : tuple[int, ...]
            shape of the list of g-space buffers

        Returns
        -------
        NDArray
            Empty buffer of shape ``(*shape, size)``
        """
        if shape == ():
            return self.create_buffer((self.size_g, ))
        if isinstance(shape, int):
            shape = (shape, )
        return self.create_buffer((*shape, self.size_g))

    def check_buffer_r(self, arr: NDArray) -> None:
        self.check_buffer(arr)
        if not (arr.ndim >= 1 and arr.shape[-1] == self.size_r):
            raise ValueError("shape of 'arr' invalid. "
                             f"expected: {(..., self.size_r)}, "
                             f"got: {arr.shape}")

    def check_buffer_g(self, arr: NDArray) -> None:
        self.check_buffer(arr)
        if not (arr.ndim >= 1 and arr.shape[-1] == self.size_g):
            raise ValueError("shape of 'arr' invalid. "
                             f"expected: {(..., self.size_g)}, "
                             f"got: {arr.shape}")

    def r2g(self, arr_r: NDArray, arr_g: Optional[NDArray] = None) -> NDArray:
        self.check_buffer_r(arr_r)
        if arr_g is not None:
            self.check_buffer_g(arr_g)
        else:
            arr_g = self.create_buffer_g(arr_r.shape[:-1])

        for inp, out in zip(arr_r.reshape(-1, *self.grid_shape),
                            arr_g.reshape(-1, self.size_g)):
            self._fft.r2g(inp, out)

        return arr_g

    def g2r(self, arr_g: NDArray, arr_r: Optional[NDArray] = None) -> NDArray:
        self.check_buffer_g(arr_g)
        if arr_r is not None:
            self.check_buffer_r(arr_r)
        else:
            arr_r = self.create_buffer_r(arr_g.shape[:-1])

        for inp, out in zip(arr_g.reshape(-1, self.size_g),
                            arr_r.reshape(-1, *self.grid_shape)):
            self._fft.g2r(inp, out)

        return arr_r
