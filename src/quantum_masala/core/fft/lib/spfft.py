from typing import Literal

import numpy as np
from scipy.fft import fftn, ifftn

from ..libwrapper import FFTLibWrapper

from quantum_masala.config import SPFFT_CONFIG
NUM_THREADS = SPFFT_CONFIG["NUM_THREADS"]


class SpFFTLibWrapper(FFTLibWrapper):
    def __init__(self, shape: tuple[int, ...], axes: tuple[int, ...]):
        super().__init__(shape, axes)

    def _execute(
        self,
        arr_in: np.ndarray,
        arr_out: np.ndarray,
        direction: Literal["forward", "backward"],
    ) -> None:
        if direction == "forward":
            arr_out[:] = fftn(arr_in, axes=self.axes, workers=NUM_THREADS)
        else:
            arr_out[:] = ifftn(arr_in, axes=self.axes, workers=NUM_THREADS)
