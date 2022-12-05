"""TODO: Complete the module
"""
from .base import FFTModule
from .backend import get_fft_backend


class FFTStick(FFTModule):

    __slots__ = ["fft"]

    def __init__(self, grid_shape, idxgrid):
        super().__init__(grid_shape, idxgrid)
        self.fft = get_fft_backend()(self.grid_shape, (0, 1, 2))

    def _g2r(self, arr_in, arr_out, overwrite_in):
        pass

    def _r2g(self, arr_in, arr_out, overwrite_in):
        pass
