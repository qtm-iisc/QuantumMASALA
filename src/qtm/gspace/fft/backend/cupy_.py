# TODO: Use CuFFT directly to implement in-place operations
# from __future__ import annotations
from typing import Union, Sequence
from qtm.config import NDArray
__all__ = ['CuPyFFTWrapper']
import numpy as np
import cupy as cp
from cupyx.scipy.fft import get_fft_plan, fftn, ifftn

from qtm.gspace.fft.backend.base import FFTBackend


class CuPyFFTWrapper(FFTBackend):

    def __init__(self, arr: Union[NDArray, Sequence[int]],
                 axes: tuple[int, ...]):
        super().__init__(arr, axes)
        arr = self.create_buffer(self.shape)
        self.fft_plan = get_fft_plan(arr, axes=self.axes, value_type='C2C')

    def set_arr(self, new_arr):
        super().set_arr(new_arr)

    @classmethod
    def create_buffer(cls, shape: tuple[int, ...]) -> cp.ndarray:
        return cp.empty(shape, dtype='c16')

    @classmethod
    def check_buffer(cls, arr: np.ndarray) -> None:
        if not isinstance(arr, cp.ndarray):
            raise TypeError(f"'arr' must be a CuPy 'ndarray'. got {type(arr)}")

    def fft(self) -> None:
        out = fftn(self.arr, axes=self.axes, plan=self.fft_plan, overwrite_x=True)
        self.arr[:] = out

    def ifft(self, normalise_idft: bool = False) -> None:
        out = ifftn(self.arr, axes=self.axes, plan=self.fft_plan, overwrite_x=True,
                    norm='backward' if normalise_idft else 'forward')
        self.arr[:] = out
