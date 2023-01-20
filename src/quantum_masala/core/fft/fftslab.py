__all__ = ['FFTSlab']
from time import perf_counter
import numpy as np

from .base import FFTModule
from .backend import get_fft_backend


class FFTSlab(FFTModule):

    __slots__ = ["fft", 'stats_r2g', 'stats_g2r']

    def __init__(self, grid_shape, idxgrid):
        super().__init__(grid_shape, idxgrid)
        self.fft = get_fft_backend()(self.grid_shape, (0, 1, 2))
        self.idxgrid = np.ravel_multi_index(self.idxgrid, grid_shape, 'C')
        self.stats_r2g = {'call': 0, 'arr': 0, 'time': 0.0, 'fft_time': 0.0}
        self.stats_g2r = {'call': 0, 'arr': 0, 'time': 0.0, 'fft_time': 0.0}

    def _g2r(self, arr_in, arr_out, overwrite_in):
        start_time = perf_counter()
        numarr = arr_in.shape[0]
        arr_out[:] = 0
        # arr_out[(slice(None), *self.idxgrid)] = arr_in

        for idxarr in range(numarr):
            arr_out[idxarr].put(self.idxgrid, arr_in[idxarr])
            self.fft.do_ifft(arr_out[idxarr])
        self.stats_g2r['fft_time'] += perf_counter() - start_time

        self.stats_g2r['call'] += 1
        self.stats_g2r['arr'] += numarr
        self.stats_g2r['time'] += perf_counter() - start_time
        return arr_out

    def _r2g(self, arr_in, arr_out, overwrite_in):
        start_time = perf_counter()
        numarr = arr_in.shape[0]
        if not overwrite_in:
            arr_in = arr_in.copy()  # Copy needed to prevent input from being overwritten

        for idxarr in range(numarr):
            self.fft.do_fft(arr_in[idxarr])
        self.stats_r2g['fft_time'] += perf_counter() - start_time

        arr_in.reshape(numarr, -1).take(self.idxgrid, 1, arr_out)
        # arr_out[:] = arr_in[(slice(None), *self.idxgrid)]
        self.stats_r2g['call'] += 1
        self.stats_r2g['arr'] += numarr
        self.stats_r2g['time'] += perf_counter() - start_time
        return arr_out
