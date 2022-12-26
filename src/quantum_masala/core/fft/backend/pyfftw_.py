import pyfftw

from quantum_masala import config
from ..base import FFTBackend


class PyFFTWLibWrapper(FFTBackend):

    __slots__ = ['fft_worker', 'ifft_worker']

    def __init__(self, shape, axes):
        super().__init__(shape, axes)
        arr = pyfftw.empty_aligned(self.shape, dtype="c16")

        fftw_flags = (config.pyfftw_planner, *config.pyfftw_flags)
        self.fft_worker = pyfftw.FFTW(arr, arr, self.axes,
                                      direction='FFTW_FORWARD',
                                      flags=fftw_flags, threads=config.fft_threads
                                      )
        self.ifft_worker = pyfftw.FFTW(arr, arr, self.axes,
                                       direction='FFTW_BACKWARD',
                                       flags=fftw_flags, threads=config.fft_threads
                                       )

    def _execute(self, arr, direction):
        # NOTE: Dont call ``FFTW.execute()`` as it will not normalize inverse
        # FFT. Just access its ``__call__()`` as it will automatically normalize
        # like in ``norm='backward'`` case
        if direction == "forward":
            self.fft_worker(arr, arr)
        else:
            self.ifft_worker(arr, arr)
