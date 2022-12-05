from scipy.fft import fftn, ifftn

from quantum_masala import config
from ..base import FFTBackend


class SpFFTLibWrapper(FFTBackend):
    def __init__(self, shape, axes):
        super().__init__(shape, axes)

    def _execute(self, arr, direction):
        if direction == "forward":
            fftn(arr, axes=self.axes, overwrite_x=True,
                 workers=config.fft_threads)
        else:
            ifftn(arr, axes=self.axes, overwrite_x=True,
                  workers=config.fft_threads)
