from typing import Literal

import numpy as np
import pyfftw

from ..libwrapper import FFTLibWrapper

from quantum_masala.config import PYFFTW_CONFIG
PLANNER_EFFORT = PYFFTW_CONFIG["PLANNER_EFFORT"]
NUM_THREADS = PYFFTW_CONFIG["NUM_THREADS"]


class PyFFTWLibWrapper(FFTLibWrapper):
    def __init__(self, shape: tuple[int, ...], axes: int):
        super().__init__(shape, axes)
        arr = pyfftw.empty_aligned(self.shape, dtype="c16")

        if self.n_axes == 1:
            fft_builder, ifft_builder = pyfftw.builders.fft, pyfftw.builders.ifft
            kwargs = {"axis": -1}
        elif self.n_axes == 2:
            fft_builder, ifft_builder = pyfftw.builders.fft2, pyfftw.builders.ifft2
            kwargs = {"axes": self.axes}
        else:
            fft_builder, ifft_builder = pyfftw.builders.fftn, pyfftw.builders.ifftn
            kwargs = {"axes": self.axes}

        kwargs["overwrite_input"] = True
        kwargs["auto_align_input"] = True
        kwargs["auto_contiguous"] = True
        kwargs["planner_effort"] = PLANNER_EFFORT
        kwargs["threads"] = NUM_THREADS

        self.fft_worker = fft_builder(arr, **kwargs)
        self.ifft_worker = ifft_builder(arr, **kwargs)

    def _execute(
        self,
        arr_in: np.ndarray,
        arr_out: np.ndarray,
        direction: Literal["forward", "backward"],
    ) -> None:
        if direction == "forward":
            self.fft_worker(arr_in, arr_out)
        else:
            self.ifft_worker(arr_in, arr_out)
