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

    @classmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return np.empty(shape, dtype='c16', order='C')

    def fft(self) -> NDArray:
        return np.fft.fftn(self._inp_fwd, axes=self.axes, norm=None)

    def ifft(self, normalise_idft: bool = False) -> NDArray:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        return np.fft.ifftn(
            self._inp_bwd, axes=self.axes,
            norm='backward' if normalise_idft else 'forward'
        )
