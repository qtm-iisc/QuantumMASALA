# from __future__ import annotations
from typing import Union, Sequence
from qtm.config import NDArray
__all__ = ['MKLFFTWrapper']

import numpy as np
import mkl_fft

from .base import FFTBackend


class MKLFFTWrapper(FFTBackend):

    def __init__(self, arr: Union[NDArray, Sequence[int]],
                 axes: tuple[int, ...]
                 ):
        super().__init__(arr, axes)
        self.unnormalize_fac = 1 / np.prod([self.shape[ai] for ai in self.axes])

    def set_arr(self, new_arr):
        super().set_arr(new_arr)

    @classmethod
    def create_buffer(cls, shape: tuple[int, ...]) -> np.ndarray:
        return np.empty(shape, dtype='c16', order='C')

    @classmethod
    def check_buffer(cls, arr: NDArray) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"'arr' must be a NumPy 'ndarray'. got {type(arr)}")

    def fft(self) -> None:
        self.arr[:] = mkl_fft.fftn(self.arr, axes=self.axes)# , norm=None)

    def ifft(self, normalise_idft: bool = False) -> None:
        self.arr[:] = mkl_fft.ifftn(
            self.arr, axes=self.axes,
            forward_scale=1 if normalise_idft else self.unnormalize_fac
        )
