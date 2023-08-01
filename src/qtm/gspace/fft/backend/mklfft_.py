
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

    @classmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return np.empty(shape, dtype='c16', order='C')

    def fft(self) -> NDArray:
        return mkl_fft.fftn(self._inp_fwd, axes=self.axes)# , norm=None)

    def ifft(self, normalise_idft: bool = False) -> NDArray:
        return mkl_fft.ifftn(
            self._inp_bwd, axes=self.axes,
            forward_scale=1 if normalise_idft else self.unnormalize_fac
        )
