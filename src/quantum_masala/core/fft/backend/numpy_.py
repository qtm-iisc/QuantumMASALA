__all__ = ['NpFFTLibWrapper']
import numpy as np

from ..base import FFTBackend


class NpFFTLibWrapper(FFTBackend):

    def __init__(self, arr: np.ndarray, axes: tuple[int, ...], normalise_idft: bool):
        super().__init__(arr, axes, normalise_idft)

    @classmethod
    def create_buffer(cls, shape: tuple[int, ...]) -> np.ndarray:
        return np.empty(shape, dtype='c16')

    def fft(self, arr: np.ndarray) -> None:
        arr[:] = np.fft.fftn(arr, axes=self.axes, norm=None)

    def ifft(self, arr: np.ndarray) -> None:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        arr[:] = np.fft.ifftn(arr, axes=self.axes,
                              norm='backward' if self.normalise_idft else 'forward')
