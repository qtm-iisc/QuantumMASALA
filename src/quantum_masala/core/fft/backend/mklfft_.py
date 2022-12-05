from mkl_fft import fftn, ifftn

from ..base import FFTBackend


class MKLFFTLibWrapper(FFTBackend):
    def __init__(self, shape, axes):
        super().__init__(shape, axes)

    def _execute(self, arr, direction):
        if direction == "forward":
            fftn(arr, axes=self.axes, overwrite_x=True)
        else:
            ifftn(arr, axes=self.axes, overwrite_x=True)
