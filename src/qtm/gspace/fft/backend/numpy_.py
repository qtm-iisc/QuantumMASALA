# from __future__ import annotations
from typing import Union, Sequence
from qtm.config import NDArray
__all__ = ['NumPyFFTWrapper']
import numpy as np

from .base import FFTBackend


class NumPyFFTWrapper(FFTBackend):

    def __init__(self, arr: Union[NDArray, Sequence[int]],
                 axes: tuple[int, ...]):
        super().__init__(arr, axes)

    def set_arr(self, new_arr):
        super().set_arr(new_arr)

    @classmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return np.empty(shape, dtype='c16', order='C')

    @classmethod
    def check_buffer(cls, arr: np.ndarray) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"'arr' must be a NumPy 'ndarray'. got {type(arr)}")

    def fft(self) -> None:
        self.arr[:] = np.fft.fftn(self.arr, axes=self.axes, norm=None)

    def ifft(self, normalise_idft: bool = False) -> None:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        self.arr[:] = np.fft.ifftn(
            self.arr, axes=self.axes,
            norm='backward' if normalise_idft else 'forward'
        )
