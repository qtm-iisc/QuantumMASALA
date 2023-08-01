# from __future__ import annotations
from typing import Union, Sequence
from qtm.config import NDArray
__all__ = ['PyFFTWFFTWrapper']
import pyfftw
import numpy as np

from qtm import qtmconfig

from .base import FFTBackend


class PyFFTWFFTWrapper(FFTBackend):

    def __init__(self, arr: Union[NDArray, Sequence[int]],
                 axes: tuple[int, ...]):
        super().__init__(arr, axes)
        self._out = self.create_buffer(self.shape)

        fftw_flags = (qtmconfig.pyfftw_planner, *qtmconfig.pyfftw_flags)
        fft_threads = qtmconfig.fft_threads

        self.plan_fw = pyfftw.FFTW(self._inp_fwd, self._out, self.axes,
                                   direction='FFTW_FORWARD',
                                   flags=fftw_flags, threads=fft_threads,
                                   )
        self.plan_bw = pyfftw.FFTW(self._inp_bwd, self._out, self.axes,
                                   direction='FFTW_BACKWARD',
                                   flags=fftw_flags, threads=fft_threads,
                                   normalise_idft=False
                                   )

    @classmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return pyfftw.empty_aligned(shape, dtype='c16', order='C')

    def fft(self) -> NDArray:
        self.plan_fw()
        return self._out

    def ifft(self, normalise_idft: bool = False) -> NDArray:
        if normalise_idft:
            self.plan_bw(normalise_idft=True)
        else:
            self.plan_bw.execute()
        return self._out
