__all__ = ['FFT3D', 'FFT3DSticks']

import numpy as np

from .base import FFTDriver
from .backend import get_fft_backend


class FFT3D(FFTDriver):

    def __init__(self, grid_shape: tuple[int, int, int],
                 idxgrid: tuple[list[int], ...], normalise_idft: bool = True):
        super().__init__(grid_shape, idxgrid, normalise_idft)

        FFT = get_fft_backend()
        self.buffer = FFT.create_buffer(self.grid_shape)
        self.fft = FFT(self.buffer, (0, 1, 2), self.normalise_idft)
        self.idxgrid = np.ravel_multi_index(self.idxgrid, self.grid_shape)

    def _g2r(self, arr_inp: np.ndarray, arr_out: np.ndarray):
        np.copyto(self.buffer, 0)
        np.put(self.buffer, self.idxgrid, arr_inp, 'wrap')
        self.fft.ifft(self.buffer)
        np.copyto(arr_out, self.buffer)

    def _r2g(self, arr_inp: np.ndarray, arr_out: np.ndarray):
        np.copyto(self.buffer, arr_inp)
        self.fft.fft(self.buffer)
        self.buffer.take(self.idxgrid, None, arr_out)


class FFT3DSticks(FFTDriver):

    def __init__(self, grid_shape: tuple[int, int, int],
                 idxgrid: tuple[list[int], ...], normalise_idft: bool = True):
        super().__init__(grid_shape, idxgrid, normalise_idft)
        nx, ny, nz = self.grid_shape
        ix, iy, iz = self.idxgrid
        idxgrid_yz = iy * nz + iz
        idxgrid_yz = np.unique(idxgrid_yz)
        self.fftx_map = [idxgrid_yz // nz, idxgrid_yz % nz]
        self.ffty_map = np.unique(self.idxgrid[0])

        FFT = get_fft_backend()
        self.buffer = FFT.create_buffer(self.grid_shape)
        self.fftx = FFT(self.buffer[:, 0, 0], (0, ))
        self.ffty = FFT(self.buffer[:, :, 0], (1, ))
        self.fftz = FFT(self.buffer[:, :, :], (2, ))

        self.idxgrid = np.ravel_multi_index(self.idxgrid, self.grid_shape)

    def _g2r(self, arr_inp: np.ndarray, arr_out: np.ndarray):
        np.copyto(self.buffer, 0)
        np.put(self.buffer, self.idxgrid, arr_inp, 'wrap')
        for iy, iz in zip(*self.fftx_map):
            self.fftx.ifft(self.buffer[:, iy, iz])
        for iz in self.ffty_map:
            self.ffty.ifft(self.buffer[:, :, iz])
        self.fftz.ifft(self.buffer[:, :, :])

    def _r2g(self, arr_inp: np.ndarray, arr_out: np.ndarray):
        np.copyto(self.buffer, arr_inp)
        self.fftz.fft(self.buffer[:, :, :])
        for iz in self.ffty_map:
            self.ffty.fft(self.buffer[:, :, iz])
        for iy, iz in zip(*self.fftx_map):
            self.fftx.fft(self.buffer[:, iy, iz])
        self.buffer.take(self.idxgrid, None, arr_out)
