# from __future__ import annotations
from qtm.typing import Union, Sequence, Type
from qtm.config import NDArray
__all__ = ['FFTBackend', ]

from abc import ABC, abstractmethod


class FFTBackend(ABC):
    """Abstract Base Class that describes a common interface to FFT Libraries.

    The FFT's in QuantumMASALA involves out-of-place FFT operations on a
    pre-defined work area. In-place FFT operationa are not optimal here due
    to the truncation of fourier space. The interface provides pre-allocated
    work arrays for input; one forward, one backward.
    A separate input buffer for backwards FFT saves in zeroing the truncated
    regions for every backward FFT.

    Parameters
    ----------
    shape: Sequence[int]
        Shape of the FFT array(s)
    axes : tuple[int, ...]
        Axes over which to compute the FFT
    """

    ndarray: Type[NDArray]

    @abstractmethod
    def __init__(self, shape: tuple[int, ...],
                 axes: tuple[int, ...]):

        self.shape: tuple[int, ...] = shape
        self.axes: tuple[int, ...] = axes
        self._inp_fwd: NDArray = self.create_buffer(self.shape)
        self._inp_bwd: NDArray = self.create_buffer(self.shape)

    @property
    def inp_fwd(self):
        return self._inp_fwd

    @property
    def inp_bwd(self):
        return self._inp_bwd

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
    def check_buffer(cls, arr: NDArray) -> None:
        """Checks if the layout of the buffer is of the right type and shape

        Parameters
        ----------
        arr : NDArray
            Input Buffer to be checked

        Raises
        ------
        ValueError
            Raised if 'arr' fails to match the type `ndarray` or array datatype
            ``'c16'``.
        """
        if not isinstance(arr, cls.ndarray):
            raise TypeError(f"'arr' must be a {cls.ndarray} instance. "
                            f"got {type(arr)}")
        if arr.dtype != 'c16':
            raise ValueError(f"'dtype' of 'arr' must be 'c16'. "
                             f"got arr.dtype = {arr.dtype}.")

    @abstractmethod
    def fft(self) -> NDArray:
        """Performs in-place forward FFT operation on `arr`
        """
        pass

    @abstractmethod
    def ifft(self, normalise_idft: bool = False) -> NDArray:
        """Performs in-place backwards FFT operation on `arr

        Parameters
        ----------
        normalise_idft : bool
            If True, the array will be divided by the product of FFT dimensions
            Else, the backwards FFT will be unnormalized
        """
        pass
