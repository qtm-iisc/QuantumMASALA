from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
__all__ = ["GSpaceBase", "cryst2idxgrid"]

import numpy as np

from qtm.lattice import ReciLattice
from qtm.fft import FFT3DFull
from qtm.msg_format import value_mismatch_msg

from qtm.constants import TPI
from qtm.config import NDArray


ROUND_PREC: int = 6
"""Rounding precision of float used when sorting.
"""


def check_g_cryst(shape: tuple[int, int, int], g_cryst: NDArray) -> None:
    """Function to validate a given array containing G-vectors in crystal
    coordinates

    The input list of G-vectors `g_cryst` must be an ``'i8'`` array of shape
    ``(3, numg)``. All vectors must lie within the FFT grid of shape `shape`
    i.e ``g_cryst[idim, :]`` must be within the range
    :math:`[-(shape[idim]//2), (shape[idim]+1)//2)`

    Parameters
    ----------
    shape : tuple[int, int, int]
        Dimensions of the FFT grid
    g_cryst : NDArray
        (``(3, -1)``, ``'i8'``) Input list of G-vectors
    """
    # Validate 'shape' param
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    assert all(isinstance(ni, int) and ni > 0 for ni in shape)

    # Check if 'g_cryst' is an array with correct shape and dtype
    from qtm.config import NDArray

    assert isinstance(
        g_cryst, NDArray
    ), f"Expected 'g_cryst' to be a {NDArray} array, got {type(g_cryst)}."
    assert g_cryst.ndim == 2
    assert g_cryst.dtype == "i8"
    assert g_cryst.shape[0] == 3

    # Check if values are within bounds
    for idim, n in enumerate(shape):
        assert np.all(g_cryst[idim] >= -(n // 2)) and np.all(
            g_cryst[idim] < (n + 1) // 2
        ), f"'g_cryst' has points that lie outside of the FFT grid"


def cryst2idxgrid(shape: tuple[int, int, int], g_cryst: NDArray) -> NDArray:
    """Converts input list of G-vectors in crystal coordinates to a 1D array of
    indices that is used to take/put corresponding values from/to a 3D grid
    with dimensions ``shape``

    Parameters
    ----------
    shape : tuple[int, int, int]
        Dimensions of the FFT Grid
    g_cryst : NDArray
        (``(3, -1)``, ``'i8'``) Input list of G-vectors. Refer to
        ``check_g_cryst``
    Returns
    -------
    idxgrid : NDArray
        A (``(g_cryst.shape[0], )``, ``'i8'``) array that can be used to index
        the corresponding values of ``g_cryst`` in a flattened 3D array with
        dimensions ``shape``
    """
    check_g_cryst(shape, g_cryst)
    n1, n2, n3 = shape
    i1, i2, i3 = g_cryst
    idxgrid = (
        n2 * n3 * (i1 + n1 * (i1 < 0))
        + n3 * (i2 + n2 * (i2 < 0))
        + (i3 + n3 * (i3 < 0))
    )
    return idxgrid


def _sort_g(grid_shape: tuple[int, int, int], idxgrid: NDArray) -> NDArray:
    """Returns the optimal sorting of G-vectors

    The optimal sort is defined based on its location on the 3D Grid.
    The points are ordered such that points lying on a parallel to the
    X Axis are grouped together. The groups are indexed by the remaining
    coordinates (y, z) and within each group, the points are sorted in
    ascending order of x coordinate.

    This is done because when distributing the 3D space across processes,
    the array must be sliced across the X-Axis so that the data local
    to each process is a contiguous chunk of the global memory.
    This requires the G-vectors to be grouped into sticks that are
    parallel to X-axis and distributed as evenly as possibly.
    """
    ix, iy, iz = np.unravel_index(idxgrid, grid_shape, order="C")
    # Arrays stacked below because CuPy's lexsort works with a single 2D array
    # and not tuple of 1d arrays
    return np.lexsort(np.stack((ix, iz, iy)))


class GSpaceBase:
    FFT3D = FFT3DFull
    _normalise_idft: bool = True

    def __init__(
        self,
        recilat: ReciLattice,
        grid_shape: tuple[int, int, int],
        g_cryst: NDArray,
        backend: str | None = None,
    ):
        self.recilat: ReciLattice = recilat
        """Reciprocal Lattice"""

        check_g_cryst(grid_shape, g_cryst)
        self.grid_shape: tuple[int, int, int] = grid_shape
        """3D FFT Mesh dimensions"""

        idxgrid = cryst2idxgrid(self.grid_shape, g_cryst)
        idxsort = _sort_g(self.grid_shape, idxgrid)

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
            self.grid_shape,
            self.idxgrid,
            normalise_idft=self._normalise_idft,
            backend=backend,
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
        return self.recilat.norm2(self.g_cryst, "cryst")

    @property
    def g_norm(self) -> NDArray:
        """(``(size, )``, ``'f8'``) Norm of G vectors."""
        return np.sqrt(self.g_norm2)

    def allocate_array(self, shape: int | Sequence[int], dtype: str = "c16") -> NDArray:
        """Returns an empty C-contiguous array of given shape and dtype.

        Parameters
        ----------
        shape : Union[int, Sequence[int]]
            Shape of the array to create.
        dtype : str, default='c16'
            String representing the array dtype to be allocated.

        Returns
        -------
        NDArray
            Empty array of given shape and dtye
        """
        return self._fft.allocate_array(shape, dtype)

    def check_array_type(self, arr: NDArray) -> None:
        """Checks if input array type is compatible with the `GSpace` instance.

        Alias of `qtm.gspace.fft.backend.FFTBackend.check_array_type`

        Parameters
        ----------
        arr : NDArray
            Input array to be validated.

        Raises
        -------
        ValueError
            Raised if input buffer fails any checks
        """
        self._fft.check_array_type(arr)

    def check_array_r(self, arr: NDArray) -> None:
        """Checks if last axis of input array has length `size_r`

        Parameters
        ----------
        arr : NDArray
            Input array to be validated.

        Raises
        -------
        ValueError
            Raised if input array fails checks
        """
        self.check_array_type(arr)
        if arr.ndim == 0:
            raise ValueError(
                value_mismatch_msg("arr.ndim", arr.ndim, "a positive integer")
            )
        if arr.shape[-1] != self.size_r:
            raise ValueError(
                value_mismatch_msg("arr.shape", arr.shape, (..., self.size_r))
            )

    def check_array_g(self, arr: NDArray) -> None:
        """Checks if last axis of input array has length `size_g`

        Parameters
        ----------
        arr : NDArray
            Input array to be validated.

        Raises
        -------
        ValueError
            Raised if input array fails checks
        """
        self.check_array_type(arr)
        if arr.ndim == 0:
            raise ValueError(
                value_mismatch_msg("arr.ndim", arr.ndim, "a positive integer")
            )
        if arr.shape[-1] != self.size_g:
            raise ValueError(
                value_mismatch_msg("arr.shape", arr.shape, (..., self.size_g))
            )

    def _r2g(self, arr_r: NDArray, arr_g: NDArray):
        for inp, out in zip(
            arr_r.reshape(-1, *self.grid_shape), arr_g.reshape(-1, self.size_g)
        ):
            self._fft.r2g(inp, out)

    def r2g(self, arr_r: NDArray, arr_g: NDArray | None = None) -> NDArray:
        self.check_array_r(arr_r)
        if arr_g is not None:
            self.check_array_g(arr_g)
        else:
            arr_g = self.allocate_array((*arr_r.shape[:-1], self.size_g))
        self._r2g(arr_r, arr_g)
        return arr_g

    def _g2r(self, arr_g: NDArray, arr_r: NDArray):
        for inp, out in zip(
            arr_g.reshape(-1, self.size_g), arr_r.reshape(-1, *self.grid_shape)
        ):
            self._fft.g2r(inp, out)

    def g2r(self, arr_g: NDArray, arr_r: NDArray | None = None) -> NDArray:
        self.check_array_g(arr_g)
        if arr_r is not None:
            self.check_array_r(arr_r)
        else:
            arr_r = self.allocate_array((*arr_g.shape[:-1], self.size_r))
        self._g2r(arr_g, arr_r)
        return arr_r
