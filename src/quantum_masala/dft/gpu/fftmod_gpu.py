from typing import Optional

import cupy as cp
from cupyx.scipy.fft import get_fft_plan
from cupyx.scipy.fftpack import fftn, ifftn

from quantum_masala.core.gspc import GkSpace
from quantum_masala.core.fft import FFTSlab, FFTBackend


class CpFFTLibWrapper(FFTBackend):

    def __init__(self, shape, axes):
        super().__init__(shape, axes)
        arr = cp.random.random(self.shape).astype('c16')
        self.fft_plan = get_fft_plan(arr, axes=self.axes, value_type='C2C')

    def _execute(self, arr, direction):
        if direction == 'forward':
            fftn(arr, axes=self.axes, plan=self.fft_plan, overwrite_x=True)
        else:
            ifftn(arr, axes=self.axes, plan=self.fft_plan, overwrite_x=True)


class CpFFTSlab(FFTSlab):

    def __init__(self, grid_shape, idxgrid):
        super().__init__(grid_shape, idxgrid)
        self.idxgrid = cp.array(self.idxgrid)
        self.fft = CpFFTLibWrapper(self.grid_shape, (0, 1, 2))

    @classmethod
    def from_gkspc(cls, gkspc: GkSpace):
        grid_shape = gkspc.gspc.grid_shape
        idxgrid = gkspc.idxgrid
        return cls(grid_shape, idxgrid)

    def g2r(self, arr_in: cp.ndarray, arr_out: Optional[cp.ndarray] = None,
            overwrite_in: bool = False):
        arr_in_ = arr_in.reshape(-1, self.numgrid)

        if arr_out is None:
            arr_out = cp.empty((*arr_in.shape[:-1], *self.grid_shape), dtype='c16')
        arr_out_ = arr_out.reshape(-1, *self.grid_shape)
        self._g2r(arr_in_, arr_out_, overwrite_in)

        return arr_out

    def r2g(self, arr_in: cp.ndarray, arr_out: Optional[cp.ndarray] = None,
            overwrite_in: bool = False):
        arr_in_ = arr_in.reshape((-1, *self.grid_shape))

        if arr_out is None:
            arr_out = cp.empty((*arr_in.shape[:-3], self.numgrid), dtype="c16")
        arr_out_ = arr_out.reshape(-1, self.numgrid)
        self._r2g(arr_in_, arr_out_, overwrite_in)

        return arr_out
