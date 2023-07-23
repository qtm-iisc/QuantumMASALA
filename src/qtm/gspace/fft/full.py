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

        self._work: NDArray = self.FFTBackend.create_buffer(self.shape)
        self.worker = self.FFTBackend(self._work, (0, 1, 2),)

    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self._work[:] = arr_inp
        self.worker.fft()
        self._work.take(self.idxgrid, out=arr_out, mode='clip')

    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self._work[:].fill(0)
        self._work.put(self.idxgrid, arr_inp)
        self.worker.ifft(self.normalise_idft)
        arr_out[:] = self._work
