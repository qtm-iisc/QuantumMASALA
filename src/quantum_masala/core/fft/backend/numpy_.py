
from numpy.fft import fftn, ifftn

from ..base import FFTBackend


class NpFFTLibWrapper(FFTBackend):
    def __init__(self, shape, axes):
        super().__init__(shape, axes)

    def _execute(self, arr, direction):
        if direction == "forward":
            arr[:] = fftn(arr, axes=self.axes)
        else:
            arr[:] = ifftn(arr, axes=self.axes)
