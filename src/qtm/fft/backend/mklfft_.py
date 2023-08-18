from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
__all__ = ['MKLFFTWrapper']

import numpy as np
import mkl_fft

from .base import FFTBackend


class MKLFFTWrapper(FFTBackend):
    """Wraps mkl_fft routines `mkl_fft.fftn` and `mkl_fft.ifftn`"""
    ndarray = np.ndarray

    def __init__(self, arr: tuple[int, ...],
                 axes: tuple[int, ...]
                 ):
        super().__init__(arr, axes)
        self.unnormalize_fac = 1 / np.prod([self.shape[ai] for ai in self.axes])

    @classmethod
    def allocate_array(cls, shape: int | Sequence[int],
                       dtype: str) -> np.ndarray:
        return np.empty(shape, dtype=dtype, order='C')

    def fft(self) -> np.ndarray:
        return mkl_fft.fftn(self._inp_fwd, axes=self.axes)# , norm=None)

    def ifft(self, normalise_idft: bool = False) -> np.ndarray:
        return mkl_fft.ifftn(
            self._inp_bwd, axes=self.axes,
            forward_scale=1 if normalise_idft else self.unnormalize_fac
        )
