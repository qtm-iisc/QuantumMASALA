# from __future__ import annotations
from typing import Union, Sequence
from qtm.config import NDArray
__all__ = ['SciPyFFTWrapper']
import numpy as np
import scipy as sp

from qtm.config import qtmconfig
from .base import FFTBackend


class SciPyFFTWrapper(FFTBackend):

    ndarray = np.ndarray

    def __init__(self, arr: tuple[int, ...],
                 axes: tuple[int, ...]):
        super().__init__(arr, axes)

    @classmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return np.empty(shape, dtype='c16', order='C')

    def fft(self) -> NDArray:
        return sp.fft.fftn(
            self._inp_fwd, axes=self.axes, norm=None, overwrite_x=False,
            workers=qtmconfig.fft_threads
        )

    def ifft(self, normalise_idft: bool = False) -> NDArray:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        return sp.fft.ifftn(
            self._inp_bwd, axes=self.axes,
            norm='backward' if normalise_idft else 'forward',
            overwrite_x=False, workers=qtmconfig.fft_threads
        )
