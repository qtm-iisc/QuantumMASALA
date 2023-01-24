__all__ = ['PyFFTWLibWrapper']
import numpy as np
import pyfftw

from quantum_masala import config
from ..base import FFTBackend


class PyFFTWLibWrapper(FFTBackend):

    def __init__(self, arr: np.ndarray, axes: tuple[int, ...], normalise_idft: bool):
        super().__init__(arr, axes, normalise_idft)

        fftw_flags = (config.pyfftw_planner, *config.pyfftw_flags)
        self.plan_fw = pyfftw.FFTW(arr, arr, self.axes,
                                   direction='FFTW_FORWARD',
                                   flags=fftw_flags, threads=config.fft_threads
                                   )
        self.plan_bw = pyfftw.FFTW(arr, arr, self.axes,
                                   direction='FFTW_BACKWARD',
                                   flags=fftw_flags, threads=config.fft_threads,
                                   normalise_idft=self.normalise_idft
                                   )

    @classmethod
    def create_buffer(cls, shape: tuple[int, ...]) -> np.ndarray:
        return pyfftw.empty_aligned(shape, dtype='c16')

    def fft(self, arr: np.ndarray) -> None:
        self.plan_fw(arr, arr)

    def ifft(self, arr: np.ndarray) -> None:
        self.plan_bw(arr, arr)
