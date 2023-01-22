__all__ = ['FFTStick']
import numpy as np
import pyfftw
from .base import FFTModule
from .backend import get_fft_backend

from quantum_masala import pw_logger


class FFTStick(FFTModule):

    __slots__ = ['num_xsticks', 'num_yplanes',
                 'fftx_map', 'fftx_shape', 'fftx', 'work_x',
                 'ffty_map', 'ffty_shape', 'ffty', 'work_y',
                 'fftz_map', 'fftz_shape', 'fftz']

    def __init__(self, grid_shape, idxgrid):
        super().__init__(grid_shape, idxgrid)

        nx, ny, nz = self.grid_shape
        ix, iy, iz = self.idxgrid
        iyz = iy * nz + iz
        isticks_yz = np.unique(iyz)
        self.num_xsticks = len(isticks_yz)

        self.fftx_shape = (nx, self.num_xsticks)
        self.fftx_map = (ix, np.searchsorted(isticks_yz, iyz))
        self.fftx_map = np.ravel_multi_index(self.fftx_map, self.fftx_shape, 'C')
        self.fftx = get_fft_backend()(self.fftx_shape, (0, ))

        isticks_y, isticks_z = isticks_yz // nz, isticks_yz % nz
        iplanes_z = np.unique(isticks_z)
        self.ffty_map = (isticks_y, np.searchsorted(iplanes_z, isticks_z))
        self.num_yplanes = len(iplanes_z)
        self.ffty_shape = (nx, ny, self.num_yplanes)
        self.ffty = get_fft_backend()(self.ffty_shape, (1, ))

        self.fftz_map = iplanes_z
        self.fftz_shape = self.grid_shape
        self.fftz = get_fft_backend()(self.fftz_shape, (2, ))

        self.work_x = pyfftw.empty_aligned(self.fftx_shape, dtype='c16')
        self.work_y = pyfftw.empty_aligned(self.ffty_shape, dtype='c16')

    def _g2r(self, arr_in, arr_out, overwrite_in):
        arr_out[:] = 0

        for inp, out in zip(arr_in, arr_out):
            self.work_x[:], self.work_y[:] = 0, 0
            self.work_x.flat[self.fftx_map] = inp
            self.fftx.do_ifft(self.work_x)
            self.work_y[(slice(None), *self.ffty_map)] = self.work_x
            self.ffty.do_ifft(self.work_y)
            out[(slice(None), slice(None), self.fftz_map)] = self.work_y
            self.fftz.do_ifft(out)

        return arr_out

    def _r2g(self, arr_in, arr_out, overwrite_in):
        if not overwrite_in:
            arr_in = arr_in.copy()

        for inp, out in zip(arr_in, arr_out):
            self.fftz.do_fft(inp)
            self.work_y[:] = inp[(slice(None), slice(None), self.fftz_map)]
            self.ffty.do_fft(self.work_y)
            self.work_x[:] = self.work_y[(slice(None), *self.ffty_map)]
            self.fftx.do_fft(self.work_x)
            out[:] = self.work_x.flat[self.fftx_map]

        return arr_out
