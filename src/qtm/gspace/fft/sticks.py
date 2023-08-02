# from __future__ import annotations
from qtm.typing import Optional
from qtm.config import NDArray
__all__ = ['FFT3DSticks']

import numpy as np

from .base import FFT3D


class FFT3DSticks(FFT3D):

    def __init__(self, shape: tuple[int, int, int],
                 idxgrid: NDArray, normalise_idft: bool,
                 backend: Optional[str] = None):
        super().__init__(shape, idxgrid, normalise_idft, backend)
        if self.idxgrid is None:
            raise Exception("'idxgrid' is None. Use 'FFT3DSlab' instead")

        idxgrid = np.unravel_index(self.idxgrid, self.shape, order='C')
        nx, ny, nz = self.shape
        ix, iy, iz = idxgrid

        iyz = iy * nz + iz
        iyz_sticks = np.unique(iyz)
        self.numsticks = len(iyz_sticks)
        iy_sticks = iyz_sticks // nz
        iz_sticks = iyz_sticks % nz

        self.g2sticks = nx * np.searchsorted(iyz_sticks, iyz) + ix
        self.fftx = self.FFTBackend((self.numsticks, nx), (1, ))
        self.fftx.inp_bwd[:] = 0

        self.sticks2full = iy_sticks * nz + iz_sticks
        self.fftyz = self.FFTBackend((nx, ny, nz), (1, 2))
        self.fftyz.inp_bwd[:] = 0

    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self.fftyz.inp_fwd[:] = arr_inp
        work_full = self.fftyz.fft().reshape((self.shape[0], -1)).T

        work_sticks = self.fftx.inp_fwd
        work_full.take(self.sticks2full, axis=0, out=work_sticks)
        work_sticks = self.fftx.fft()
        work_sticks.take(self.g2sticks, out=arr_out, mode='clip')

    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Performance reduction here due to the way we fill the array
        # Ideally, we traverse the work arrays once, filling it with
        # input values wherever necessary and rest we zero out (as it is not
        # already)
        # Instead, we have to first zero out the entire array and then
        # fill it at specific sites with values. Resulting in double traversal
        # This is where we are losing all our theoretical performance gains
        # when performing sticks FFT.

        self.fftx.inp_bwd.reshape(-1)[self.g2sticks] = arr_inp
        work_sticks = self.fftx.ifft(self.normalise_idft)

        work_full = self.fftyz.inp_bwd.reshape((self.shape[0], -1)).T
        work_full[self.sticks2full] = work_sticks
        arr_out[:] = self.fftyz.ifft(self.normalise_idft)
