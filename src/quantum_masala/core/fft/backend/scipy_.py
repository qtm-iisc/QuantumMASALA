__all__ = ['SpFFTLibWrapper']
import numpy as np
from scipy.fft import fftn, ifftn

from quantum_masala import config
from ..base import FFTBackend


class SpFFTLibWrapper(FFTBackend):

    def __init__(self, arr: np.ndarray, axes: tuple[int, ...], normalise_idft: bool):
        super().__init__(arr, axes, normalise_idft)

    @classmethod
    def create_buffer(cls, shape: tuple[int, ...]) -> np.ndarray:
        return np.empty(shape, dtype='c16')

    def fft(self, arr: np.ndarray) -> None:
        arr[:] = fftn(arr, axes=self.axes, norm=None,
                      overwrite_x=True, workers=config.fft_threads)

    def ifft(self, arr: np.ndarray) -> None:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        arr[:] = ifftn(arr, axes=self.axes,
                       norm='backward' if self.normalise_idft else 'forward',
                       overwrite_x=True, workers=config.fft_threads)
