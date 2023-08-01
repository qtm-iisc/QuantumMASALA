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

    @classmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> cp.ndarray:
        return cp.empty(shape, dtype='c16')

    def fft(self) -> NDArray:
        return fftn(self._inp_fwd, axes=self.axes, plan=self.fft_plan,
                    overwrite_x=False)

    def ifft(self, normalise_idft: bool = False) -> NDArray:
        return ifftn(self._inp_bwd, axes=self.axes, plan=self.fft_plan,
                     overwrite_x=True, norm='backward' if normalise_idft else 'forward')
