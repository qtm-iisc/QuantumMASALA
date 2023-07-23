# from __future__ import annotations
from typing import Optional
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
        self.g2sticks = ix * self.numsticks + np.searchsorted(iyz_sticks, iyz)

        self.sticks2full = iy_sticks * nz + iz_sticks
        self.work_sticks = self.FFTBackend.create_buffer((nx, self.numsticks))
        self.fftx = self.FFTBackend(self.work_sticks, (0, ))

        self.work_full = self.FFTBackend.create_buffer((nx, ny, nz))
        self.fftyz = self.FFTBackend(self.work_full, (1, 2))

    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self.work_full[:] = arr_inp
        self.fftyz.fft()

        work_full = self.work_full.reshape((self.shape[0], -1))
        work_full.take(self.sticks2full, axis=1, out=self.work_sticks, mode='clip')
        self.fftx.fft()

        self.work_sticks.take(self.g2sticks, out=arr_out, mode='clip')

    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Performance reduction here due to the way we fill the array
        # Ideally, we traverse the work arrays once, filling it with
        # input values wherever necessary and rest we zero out (as it is not
        # already)
        # Instead, we have to first zero out the entire array and then
        # fill it at specific sites with values. Resulting in double traversal
        # This is where we are losing all our theoretical performance gains
        # when performing sticks FFT.
        self.work_sticks.fill(0)
        self.work_sticks.reshape(-1)[self.g2sticks] = arr_inp
        self.fftx.ifft(self.normalise_idft)

        self.work_full.fill(0)
        self.work_full.reshape((self.shape[0], -1))[
            (slice(None), self.sticks2full)
        ] = self.work_sticks
        self.fftyz.ifft(self.normalise_idft)

        arr_out[:] = self.work_full
