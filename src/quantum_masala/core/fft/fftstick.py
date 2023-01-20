__all__ = ['FFTStick']
import numpy as np
from .base import FFTModule
from .backend import get_fft_backend


class FFTStick(FFTModule):

    __slots__ = ['numxy', 'xy_shape', 'xy_map', 'xyz_map',
                 'fft_z', 'fft_xy', 'work']

    def __init__(self, grid_shape, idxgrid):
        super().__init__(grid_shape, idxgrid)

        idxgrid_xy = self.idxgrid[0] * grid_shape[1] + self.idxgrid[1]
        l_idx_xy = np.unique(idxgrid_xy)
        xy_map = np.searchsorted(l_idx_xy, idxgrid_xy)
        self.numxy = len(l_idx_xy)
        self.xy_shape = (self.numxy, self.grid_shape[2])

        self.xy_map = (xy_map, self.idxgrid[2])
        self.xy_map = np.ravel_multi_index(self.xy_map, self.xy_shape, 'C')

        self.xyz_map = np.unravel_index(
            np.arange(np.prod(self.xy_shape), dtype='i8'), self.xy_shape, 'C'
        )

        self.xyz_map = (l_idx_xy[self.xyz_map[0]], self.xyz_map[1])
        self.xyz_map = (self.xyz_map[0] // self.grid_shape[1],
                        self.xyz_map[0] % self.grid_shape[1],
                        self.xyz_map[1])
        self.xyz_map = np.ravel_multi_index(self.xyz_map, self.grid_shape, 'C')

        self.fft_z = get_fft_backend()(self.xy_shape, (1,))
        self.fft_xy = get_fft_backend()(self.grid_shape, (0, 1))
        self.work = np.empty(self.xy_shape, dtype='c16', order='C')

    def _g2r(self, arr_in, arr_out, overwrite_in):
        numarr = arr_in.shape[0]
        arr_out[:] = 0

        for idxarr in range(numarr):
            self.work[:] = 0
            self.work.put(self.xy_map, arr_in[idxarr])
            self.fft_z.do_ifft(self.work)
            arr_out[idxarr].put(self.xyz_map, self.work, 'raise')
            self.fft_xy.do_ifft(arr_out[idxarr])

        return arr_out

    def _r2g(self, arr_in, arr_out, overwrite_in):
        numarr = arr_in.shape[0]
        if not overwrite_in:
            arr_in = arr_in.copy()  # Copy needed to prevent input from being overwritten

        for idxarr in range(numarr):
            self.fft_xy.do_fft(arr_in[idxarr])
            arr_in[idxarr].reshape(-1).take(
                self.xyz_map, None, self.work.reshape(-1))
            self.fft_z.do_fft(self.work)
            self.work.take(self.xy_map, None, arr_out[idxarr])

        return arr_out
