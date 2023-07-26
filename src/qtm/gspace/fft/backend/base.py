# from __future__ import annotations
from typing import Union, Sequence, Optional
from qtm.config import NDArray
__all__ = ['FFTBackend', ]

from abc import ABC, abstractmethod


class FFTBackend(ABC):
    """Abstract Base Class that describes the interface to FFT Libraries

    Specifies a common interface to FFT libraries. Apart from FFT Routines,
    the class also provides array creation methods, allowing implementation
    of routines that create optimal buffers for said FFT routines.

    Parameters
    ----------
    arr : Union[NDArray, Sequence[int]]
        Input FFT Array for setting shapes and strides. Alternatively, it can
        also be a sequence of integers which will be interpreted as the shape
        of a C-contiguous array
    axes : tuple[int, ...]
        Axes over which to compute the FFT

    """

    @abstractmethod
    def __init__(self, arr: Union[NDArray, Sequence[int]],
                 axes: tuple[int, ...]):
        # Validating input 'arr'
        if isinstance(arr, Sequence):
            arr = self.create_buffer(arr)
        self.check_buffer(arr)

        self._arr: NDArray = arr
        """Array to operate FFT on"""
        self.shape: tuple[int, ...] = arr.shape
        """Shape of the FFT mesh"""
        self.ndim: int = arr.ndim
        """Dimension of the mesh"""
        self.strides: tuple[int, ...] = arr.strides
        """Strides of data in memory"""
        self.axes: tuple[int, ...] = axes
        """Axes over which to compute the FFT"""

    @property
    def arr(self):
        return self._arr

    @arr.setter
    def arr(self, new_arr):
        self.set_arr(new_arr)

    @abstractmethod
    def set_arr(self, new_arr):
        self._check_arr(new_arr)
        self._arr = new_arr

    def _check_arr(self, new_arr):
        self.check_buffer(new_arr)
        if new_arr.shape != self.shape:
            raise ValueError("'arr' failed to match 'shape' attribute.")
        if new_arr.strides != self.strides:
            raise ValueError("'arr' failed to match 'strides' attribute.")

    @classmethod
    @abstractmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> NDArray:
        """Create a C-contiguous complex128 buffer compatible for FFT Transforms

        Note that the atrribute 'stride' has no impact on this method. The
        array returned will always be complex128 and C-contiguous

        Parameters
        ----------
        shape : tuple[int, ...]
            Shape of the buffer

        Returns
        -------
        NDArray
            A C-contiguous ``'c16'`` array of given shape.
        """
        pass

    @classmethod
    @abstractmethod
    def check_buffer(cls, arr: NDArray) -> None:
        """Checks if the layout of the buffer is compatible for in-place FFT
        transforms (contiguous, aligned, etc.)

        Parameters
        ----------
        arr : NDArray
            Input Buffer to be checked

        Raises
        ------
        ValueError
            Raised if 'arr' is incompatible
        """
        pass

    @abstractmethod
    def fft(self) -> None:
        """Performs in-place forward FFT operation on `arr`
        """
        pass

    @abstractmethod
    def ifft(self, normalise_idft: bool = False) -> None:
        """Performs in-place backwards FFT operation on `arr

        Parameters
        ----------
        normalise_idft : bool
            If True, the array will be divided by the product of FFT dimensions
            Else, the backwards FFT will be unnormalized
        """
        pass