from typing import Optional

import cupy as cp
from cupyx.scipy.fft import get_fft_plan, fftn, ifftn

from quantum_masala.core.gspc import GkSpace
from quantum_masala.core.fft.base import FFTDriver, FFTBackend


class CpFFTLibWrapper(FFTBackend):

    def __init__(self, arr: cp.ndarray, axes: tuple[int, ...], normalize_idft: bool):
        super().__init__(arr, axes, normalize_idft)
        self.plan_fw = get_fft_plan(arr, axes=self.axes, value_type='C2C')
        self.plan_bw = self.plan_fw

    @classmethod
    def create_buffer(cls, shape: tuple[int, ...]):
        return cp.empty(shape, dtype='c16')

    def fft(self, arr: cp.ndarray):
        fftn(arr, axes=self.axes, plan=self.plan_fw, overwrite_x=True)

    def ifft(self, arr: cp.ndarray):
        ifftn(arr, axes=self.axes, plan=self.plan_bw, overwrite_x=True,
              norm='forward')


class CpFFT3D(FFTDriver):

    def __init__(self, grid_shape: tuple[int, int, int],
                 idxgrid: tuple[list[int], ...],
                 normalize_idft: bool = True):
        super().__init__(grid_shape, idxgrid, normalize_idft)
        self.idxgrid = cp.array(self.idxgrid)
        self.fft = CpFFTLibWrapper(CpFFTLibWrapper.create_buffer(self.grid_shape),
                                   (0, 1, 2), self.normalise_idft
                                   )
        self.idxgrid = cp.ravel_multi_index(self.idxgrid, self.grid_shape)

    def _g2r(self, arr_inp: cp.ndarray, arr_out: cp.ndarray):
        cp.copyto(arr_out, 0)
        cp.put(arr_out, self.idxgrid, arr_inp, 'wrap')
        self.fft.ifft(arr_out)

    def _r2g(self, arr_inp: cp.ndarray, arr_out: cp.ndarray):
        self.fft.fft(arr_inp)
        arr_inp.take(self.idxgrid, None, arr_out)
