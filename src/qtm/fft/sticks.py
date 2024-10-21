from __future__ import annotations

__all__ = ["FFT3DSticks"]

import numpy as np

from .base import FFT3D
from qtm.config import NDArray


class FFT3DSticks(FFT3D):
    """Provides FFT routines that involve skipping FFT operations along 1D
    'sticks' that are not within the list of points in the 3D grid, `idxgrid`.
    This can result in faster FFT operations in a truncated Fourier Space
    that is compact (like for instance a sphere of points within the
    3D box)
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        idxgrid: NDArray,
        normalise_idft: bool,
        backend: str | None = None,
        **kwargs,
    ):
        super().__init__(shape, idxgrid, normalise_idft, backend, **kwargs)
        if self.idxgrid is None:
            raise Exception("'idxgrid' is None. Use 'FFT3DSlab' instead")

        # Getting the x, y, z indices from flattened indices
        idxgrid = np.unravel_index(self.idxgrid, self.shape, order="C")
        nx, ny, nz = self.shape
        ix, iy, iz = idxgrid

        # The (y, z) coordinates of all the sticks are determined
        iyz = iy * nz + iz
        iyz_sticks = np.unique(iyz)
        self.numsticks = len(iyz_sticks)
        iy_sticks = iyz_sticks // nz
        iz_sticks = iyz_sticks % nz

        # Mapping from the input G-vector to the 2D array holding the list
        # of sticks is generated
        self.g2sticks = nx * np.searchsorted(iyz_sticks, iyz) + ix
        # FFT is performed along the length of the stick, which corresponds
        # to 3D arrays FFT along X-Axis
        self.fftx = self.FFTBackend((self.numsticks, nx), (1,))
        # Zeroing out ifft input array after initializing
        self.fftx.inp_bwd = 0
        # Points lying outside 'g2sticks' will not be accessed and thus will
        # remain zero throughout the instance's lifetime

        # Mapping from the sticks to the final 3D FFT array
        self.sticks2full = iy_sticks * nz + iz_sticks
        # FFT is performed along Y and Z directions now
        self.fftyz = self.FFTBackend((nx, ny, nz), (1, 2))
        # Zeroing out ifft input array after initializing
        self.fftyz.inp_bwd = 0

    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self.fftyz.inp_fwd[:] = arr_inp
        work_full = self.fftyz.fft().reshape((self.shape[0], -1)).T

        work_sticks = self.fftx.inp_fwd
        work_full.take(self.sticks2full, axis=0, out=work_sticks)
        work_sticks = self.fftx.fft()
        work_sticks.take(self.g2sticks, out=arr_out)

    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self.fftx.inp_bwd.fill(0.0)
        self.fftyz.inp_bwd.fill(0.0)
        self.fftx.inp_bwd.reshape(-1)[self.g2sticks] = arr_inp
        work_sticks = self.fftx.ifft(self.normalise_idft)

        work_full = self.fftyz.inp_bwd.reshape((self.shape[0], -1)).T
        work_full[self.sticks2full] = work_sticks
        arr_out[:] = self.fftyz.ifft(self.normalise_idft)
