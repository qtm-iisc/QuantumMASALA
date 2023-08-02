# from __future__ import annotations
from typing import Optional
from qtm.config import NDArray
__all__ = ['FFT3DFull', ]

from .base import FFT3D


class FFT3DFull(FFT3D):

    def __init__(self, shape: tuple[int, int, int],
                 idxgrid: NDArray, normalise_idft: bool,
                 backend: Optional[str] = None):
        super().__init__(shape, idxgrid, normalise_idft, backend)
        self.worker = self.FFTBackend(self.shape, (0, 1, 2),)
        self.worker.inp_bwd[:] = 0

    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self.worker.inp_fwd[:] = arr_inp
        out = self.worker.fft()
        out.take(self.idxgrid, out=arr_out, mode='clip')

    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self.worker.inp_bwd.put(self.idxgrid, arr_inp)
        arr_out[:] = self.worker.ifft(self.normalise_idft)