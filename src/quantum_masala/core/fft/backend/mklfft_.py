__all__ = ['MKLFFTLibWrapper']
import numpy as np
from mkl_fft import fftn, ifftn

from ..base import FFTBackend
from quantum_masala import config


class MKLFFTLibWrapper(FFTBackend):

    def __init__(self, arr: np.ndarray, axes: tuple[int, ...], normalise_idft: bool):
        super().__init__(arr, axes, normalise_idft)
        if not self.normalise_idft:
            self.forward_scale = 1 / np.prod([self.shape[ai] for ai in self.axes])
        else:
            self.forward_scale = 1
    @classmethod
    def create_buffer(cls, shape: tuple[int, ...]) -> np.ndarray:
        return np.empty(shape, dtype='c16')

    def fft(self, arr: np.ndarray) -> None:
        arr[:] = fftn(arr, axes=self.axes)# , norm=None)

    def ifft(self, arr: np.ndarray) -> None:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        arr[:] = ifftn(arr, axes=self.axes, forward_scale=self.forward_scale)
