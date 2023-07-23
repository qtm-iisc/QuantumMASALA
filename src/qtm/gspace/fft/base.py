# from __future__ import annotations
from typing import Type, Optional
from qtm.config import NDArray
__all__ = ['FFT3D', 'get_FFTBackend']

from abc import ABC, abstractmethod
import numpy as np

from .backend import get_FFTBackend
from .utils import check_g_idxgrid

from qtm.config import qtmconfig


class FFT3D(ABC):
    """Abstract Base Class for the FFT Module in ``GSpace``

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the 3D FFT grid
    idxgrid : NDArray
        (Flat) Indices of 3D FFT meshgrid that are evaluated; the rest are
        considered to be zeros and discarded.
    normalise_idft : bool
        If True, the inverse FFT will be normalised by ``1 / prod(shape)``
    """

    @abstractmethod
    def __init__(self, shape: tuple[int, int, int],
                 idxgrid: NDArray, normalise_idft: bool,
                 backend: Optional[str]):
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
        self.FFTBackend = get_FFTBackend(backend)

    @abstractmethod
    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        """Performs forward FFT operation

        Parameters
        ----------
        arr_inp : NDArray
            Input Array
        arr_out : NDArray
            Output Array
        """
        pass

    @abstractmethod
    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        """Performs backwards FFT operation

        Parameters
        ----------
        arr_inp : NDArray
            Input Array
        arr_out : NDArray
            Output Array
        """
        pass

    def create_buffer(self, shape: tuple[int, ...]) -> NDArray:
        """Alias of ``FFTBackend.create_buffer(shape)``"""
        return self.FFTBackend.create_buffer(shape)

    def check_buffer(self, arr: NDArray) -> None:
        """Alias of ``FFTBackend.check_buffer(arr)``"""
        self.FFTBackend.check_buffer(arr)



