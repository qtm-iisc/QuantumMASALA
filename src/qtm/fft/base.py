from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
__all__ = ['FFT3D', 'DummyFFT3D']

from abc import ABC, abstractmethod
import numpy as np

from .backend import get_FFTBackend
from qtm.config import qtmconfig

from qtm.config import NDArray


def check_g_idxgrid(shape: tuple[int, int, int], idxgrid: NDArray, check_len:bool=True):
    """Function to validate `idxgrid`, the flattened indices corresponding
    to the G-vectors in a 3D FFT array with dimensions ``shape``

    `idxgrid` must be a ``'i8'`` 1D array. All values must be unique,
     non-negative and bounded by the prodcut of `shape`. This ensures
     indices are within bounds.
    """
    assert isinstance(idxgrid, NDArray)
    assert idxgrid.ndim == 1
    assert idxgrid.dtype == 'i8'
    if check_len:
        assert len(idxgrid) == len(np.unique(idxgrid)), f"idxgrid {idxgrid}"
    assert np.all(idxgrid >= 0) and np.all(idxgrid < np.prod(shape))


class FFT3D(ABC):
    """FFT Module for transforming between real-space and the G-Space defined
    by a `GSpaceBase` instance

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the 3D FFT grid
    idxgrid : NDArray | None
        (Flat) Indices of 3D FFT meshgrid that are evaluated; the rest are
        considered to be zeros and discarded. If None, no points are discarded.
    normalise_idft : bool
        If True, the inverse FFT will be normalised by ``1 / prod(shape)``

    """

    @abstractmethod
    def __init__(self, shape: tuple[int, int, int],
                 idxgrid: NDArray | None, normalise_idft: bool,
                 backend: str | None, skip_check_g_idxgrid_len: bool = False):
        if idxgrid is None:
            idxgrid = np.arange(np.prod(shape), dtype='i8')
        if not skip_check_g_idxgrid_len:
            check_g_idxgrid(shape, idxgrid)
        self.shape: tuple[int, int, int] = shape
        """shape of the FFT grid"""

        self.idxgrid: NDArray = idxgrid
        """(``(size, )``, ``'i8'``) (Flat) Indices of 3D FFT meshgrid that are 
        evaluated; the rest are considered to be zeros and discarded."""

        if not isinstance(normalise_idft, bool):
            raise ValueError(f"'normalise_idft' must be a boolean. "
                             f"got {type(normalise_idft)}.")
        self.normalise_idft: bool = normalise_idft
        """If True, the inverse FFT will be normalised by ``1 / prod(shape)``"""
        self.normalise_fac: float = 1. / np.prod(self.shape)
        """``1. / prod(shape)``; Normalisation factor applied to backwards FFT"""

        if backend is None:
            backend = qtmconfig.fft_backend
        self.backend = backend
        self.FFTBackend = get_FFTBackend(backend)

    @abstractmethod
    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        """Performs forward FFT operation of input data restricted to only FFT
        components located in (flattened) indices `idxgrid`

        Parameters
        ----------
        arr_inp : NDArray
            (`shape`) Input Array
        arr_out : NDArray
            (`len(idxgrid`) Output Array
        """
        pass

    @abstractmethod
    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        """Performs backwards FFT operation of input data which is interpreted
        to be the only nonzero FFT components in the 3D grid and are located
        at (flattened) indices `idxgrid`

        Parameters
        ----------
        arr_inp : NDArray
            (`len(idxgrid)`) Input Array
        arr_out : NDArray
            (`shape`) Output Array
        """
        pass

    def allocate_array(self, shape: int | Sequence[int],
                       dtype: str) -> NDArray:
        """Alias of `FFTBackend.allocate_array`"""
        return self.FFTBackend.allocate_array(shape, dtype)

    def check_array_type(self, arr: NDArray) -> None:
        """Alias of `FFTBackend.check_array_type`"""
        self.FFTBackend.check_array_type(arr)


class DummyFFT3D:
    """A Dummy `FFT3D` class used as a placeholder for custom 'GSpaceBase'
    subclasses, allowing one to initialise the FFT object after calling
    the parent class constructor `GSpaceBase.__init__`

    """

    def __init__(self, *args, **kwargs):
        pass
