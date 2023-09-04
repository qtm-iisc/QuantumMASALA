from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Type
__all__ = ['PyFFTWFFTWrapper']
import pyfftw
import numpy as np

from qtm import qtmconfig

from .base import FFTBackend


class PyFFTWFFTWrapper(FFTBackend):
    """Wraps PyFFTW's FFT routines. Involves generating FFTW Plan objects
    `pyfftw.FFTW`"""

    ndarray = np.ndarray

    def __init__(self, arr: tuple[int, ...],
                 axes: tuple[int, ...]):
        super().__init__(arr, axes)
        self._out = self.allocate_array(self.shape, 'c16')

        fftw_flags = (qtmconfig.pyfftw_planner, *qtmconfig.pyfftw_flags)
        if 'FFTW_DESTROY_INPUT' in fftw_flags:
            raise ValueError("FFTW flag 'FFTW_DESTROY_INPUT' is not supported"
                             "by QTM. Please remove the flag from "
                             "'qtm.qtmconfig.pyfftw_flags' which is set to: "
                             f"{qtmconfig.pyfftw_flags}")
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
    def allocate_array(cls, shape: int | Sequence[int],
                       dtype: str) -> np.ndarray:
        return pyfftw.empty_aligned(shape, dtype=dtype, order='C')

    @property
    def out(self) -> np.ndarray:
        return self._out

    def fft(self) -> np.ndarray:
        self.plan_fw()
        return self.out

    def ifft(self, normalise_idft: bool = False) -> np.ndarray:
        if normalise_idft:
            self.plan_bw(normalise_idft=True)
        else:
            self.plan_bw.execute()
        return self.out
