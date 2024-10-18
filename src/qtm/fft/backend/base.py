from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
__all__ = [
    "FFTBackend",
]

from abc import ABC, abstractmethod
from qtm.config import NDArray


class FFTBackend(ABC):
    """Abstract Class describeing a common interface to FFT Libraries.

    Wraps N-dimensional FFT routines defined in libraries such as
    NumPy (`numpy.fft`), PyFFTW (`pyfftw`), CuPy (`cupy.fft`), etc.

    It also provides methods to allocate N-dimensional arrays of given shape
    and dtype. This allows us to write routines that do not require import
    logic based on array type, which is most commonly encountered when
    implementing CPU-GPU Agnostic routines.

    Parameters
    ----------
    shape: Sequence[int]
        Shape of the FFT array(s)
    axes : Sequence[int]
        Axes over which to compute the FFT

    Notes
    -----
    The FFT's in QuantumMASALA involves out-of-place FFT operations on a
    pre-defined work area. In-place FFT operationa are not optimal here due
    to the truncation of fourier space. The interface provides pre-allocated
    work arrays for input; `inp_fwd` for forward, `inp_bwd` for backward.
    A separate input array for backwards FFT saves in zeroing the truncated
    regions for every backward FFT. When initialized, input arrays are not
    set to zero when initialized.
    """

    ndarray: type[NDArray]
    """type of the ndarray the backend supports; used for type checking"""

    @abstractmethod
    def __init__(self, shape: Sequence[int], axes: Sequence[int]):
        shape = tuple(shape)
        assert all(isinstance(ni, int) and ni > 0 for ni in shape)
        self.shape: tuple[int, ...] = shape
        """Shape of the input/output array"""

        axes = tuple(axes)
        assert all(isinstance(ax, int) for ax in axes)
        self.axes: tuple[int, ...] = axes
        """Axes over which to perform the FFT operations"""

        self._inp_fwd: NDArray = self.allocate_array(self.shape, "c16")
        self._inp_bwd: NDArray = self.allocate_array(self.shape, "c16")

    @property
    def inp_fwd(self) -> NDArray:
        """Preallocated ``'c16'`` array of shape `shape` bound to the input
        of forward FFT operation"""
        return self._inp_fwd

    @inp_fwd.setter
    def inp_fwd(self, data: NDArray):
        self._inp_fwd[:] = data

    @property
    def inp_bwd(self) -> NDArray:
        """Preallocated ``'c16'`` array of shape `shape` bound to the input
        of backwards FFT operation"""
        return self._inp_bwd

    @inp_bwd.setter
    def inp_bwd(self, data):
        self._inp_bwd[:] = data

    @classmethod
    def allocate_array(cls, shape: int | Sequence[int], dtype: str) -> NDArray:
        """Create a C-contiguous empty array with given datatype

        Parameters
        ----------
        shape : int | Sequence[int]
            Shape of the buffer.
        dtype : str
            String representing the array dtype to be allocated.

        Returns
        -------
        NDArray
            A C-contiguous empty array of given shape and dtype.
        """
        pass

    @classmethod
    def check_array_type(cls, arr: NDArray) -> None:
        """Checks if the array type is supported

        Parameters
        ----------
        arr : NDArray
            Input array to validate

        Raises
        ------
        ValueError
            Raised if 'arr' fails to match the type `ndarray`.
        """
        if not isinstance(arr, cls.ndarray):
            raise TypeError(
                f"'arr' must be a {cls.ndarray} instance. " f"got {type(arr)}"
            )

    @abstractmethod
    def fft(self) -> NDArray:
        """Computes forward FFT operation on data in `inp_fwd` and returns the
        resulting array
        """
        pass

    @abstractmethod
    def ifft(self, normalise_idft: bool = False) -> NDArray:
        """Computes backwards FFT operation on data in `inp_bwd` and returns
        the resulting array

        Parameters
        ----------
        normalise_idft : bool
            If True, the array will be divided by the product of FFT
            dimensions. Else, the backwards FFT will be unnormalized.
        """
        pass
