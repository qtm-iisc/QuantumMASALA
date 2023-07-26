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

        fftw_flags = (qtmconfig.pyfftw_planner, *qtmconfig.pyfftw_flags)
        fft_threads = qtmconfig.fft_threads

        if not isinstance(arr, np.ndarray):
            arr = self.create_buffer(self.shape)
            
        self.plan_fw = pyfftw.FFTW(arr, arr, self.axes,
                                   direction='FFTW_FORWARD',
                                   flags=fftw_flags, threads=fft_threads,
                                   )
        self.plan_bw = pyfftw.FFTW(arr, arr, self.axes,
                                   direction='FFTW_BACKWARD',
                                   flags=fftw_flags, threads=fft_threads,
                                   normalise_idft=False
                                   )

    def set_arr(self, new_arr):
        super().set_arr(new_arr)
        self.plan_fw.update_arrays(new_arr, new_arr)
        self.plan_bw.update_arrays(new_arr, new_arr)

    @classmethod
    def create_buffer(cls, shape: Union[int, Sequence[int]]) -> np.ndarray:
        return pyfftw.empty_aligned(shape, dtype='c16', order='C')

    @classmethod
    def check_buffer(cls, arr: np.ndarray) -> None:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"'arr' must be a NumPy 'ndarray'. got {type(arr)}")
        # if not pyfftw.is_byte_aligned(arr):
        #     raise ValueError(f"'arr' failed PyFFTW's byte align check")

    def fft(self) -> None:
        self.plan_fw()

    def ifft(self, normalise_idft: bool = False) -> None:
        if normalise_idft:
            self.plan_bw(normalise_idft=True)
        else:
            self.plan_bw.execute()
