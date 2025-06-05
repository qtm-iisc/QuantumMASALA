from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
__all__ = ["SciPyFFTWrapper"]
import numpy as np
import scipy

from qtm.config import qtmconfig
from .base import FFTBackend


class SciPyFFTWrapper(FFTBackend):
    """Wraps SciPy's FFT routines, `scipy.fft.fftn` and `scipy.fft.ifftn`"""

    ndarray = np.ndarray

    def __init__(self, arr: tuple[int, ...], axes: tuple[int, ...]):
        super().__init__(arr, axes)

    @classmethod
    def allocate_array(cls, shape: int | Sequence[int], dtype: str) -> np.ndarray:
        return np.empty(shape, dtype=dtype, order="C")

    def fft(self) -> np.ndarray:
        return scipy.fft.fftn(
            self._inp_fwd,
            axes=self.axes,
            norm=None,
            overwrite_x=False,
            workers=qtmconfig.fft_threads,
        )

    def ifft(self, normalise_idft: bool = False) -> np.ndarray:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        return scipy.fft.ifftn(
            self._inp_bwd,
            axes=self.axes,
            norm="backward" if normalise_idft else "forward",
            overwrite_x=False,
            workers=qtmconfig.fft_threads,
        )
