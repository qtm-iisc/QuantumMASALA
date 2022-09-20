from typing import Literal

import numpy as np

from ..libwrapper import FFTLibWrapper


class NpFFTLibWrapper(FFTLibWrapper):
    def __init__(self, shape: tuple[int, ...], axes: tuple[int, ...]):
        super().__init__(shape, axes)

    def _execute(
        self,
        arr_in: np.ndarray,
        arr_out: np.ndarray,
        direction: Literal["forward", "backward"],
    ) -> None:
        if direction == "forward":
            arr_out[:] = np.fft.fftn(arr_in, axes=self.axes)
        else:
            arr_out[:] = np.fft.ifftn(arr_in, axes=self.axes)
