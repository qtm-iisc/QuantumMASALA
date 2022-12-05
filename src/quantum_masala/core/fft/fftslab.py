from .base import FFTModule
from .backend import get_fft_backend


class FFTSlab(FFTModule):

    __slots__ = ["fft"]

    def __init__(self, grid_shape, idxgrid):
        super().__init__(grid_shape, idxgrid)
        self.fft = get_fft_backend()(self.grid_shape, (0, 1, 2))

    def _g2r(self, arr_in, arr_out, overwrite_in):
        numarr = arr_in.shape[0]
        arr_out[:] = 0
        arr_out[(slice(None), *self.idxgrid)] = arr_in

        for idxarr in range(numarr):
            self.fft.do_ifft(arr_out[idxarr])
        return arr_out

    def _r2g(self, arr_in, arr_out, overwrite_in):
        numarr = arr_in.shape[0]
        if not overwrite_in:
            arr_in = arr_in.copy()  # Copy needed to prevent input from being overwritten

        for idxarr in range(numarr):
            self.fft.do_fft(arr_in[idxarr])

        arr_out[:] = arr_in[(slice(None), *self.idxgrid)]
        return arr_out
