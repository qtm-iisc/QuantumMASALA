from typing import Literal

import numpy as np
from mkl_fft import fftn, ifftn

from ..libwrapper import FFTLibWrapper


class MKLFFTLibWrapper(FFTLibWrapper):
    def __init__(self, shape: tuple[int, ...], axes: int):
        super().__init__(shape, axes)

    def _execute(
        self,
        arr_in: np.ndarray,
        arr_out: np.ndarray,
        direction: Literal["forward", "backward"],
    ) -> None:
        arr_out[:] = arr_in
        if direction == "forward":
            fftn(arr_out, axes=(-1, -2, -3), overwrite_x=True)
        else:
            ifftn(arr_out, axes=(-1, -2, -3), overwrite_x=True)
