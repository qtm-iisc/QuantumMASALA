from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
__all__ = ["NumPyFFTWrapper"]
import numpy as np

from .base import FFTBackend


class NumPyFFTWrapper(FFTBackend):
    """Wraps NumPy's FFT routines, `numpy.fft.fftn` and `numpy.fft.ifftn`"""

    ndarray = np.ndarray

    def __init__(self, arr: tuple[int, ...], axes: tuple[int, ...]):
        super().__init__(arr, axes)

    @classmethod
    def allocate_array(cls, shape: int | Sequence[int], dtype: str) -> np.ndarray:
        return np.empty(shape, dtype=dtype, order="C")

    def fft(self) -> np.ndarray:
        return np.fft.fftn(self._inp_fwd, axes=self.axes, norm=None)

    def ifft(self, normalise_idft: bool = False) -> np.ndarray:
        # NOTE: `norm` in NumPy as SciPy is different from pyFFTW's `normalise_idft`
        return np.fft.ifftn(
            self._inp_bwd,
            axes=self.axes,
            norm="backward" if normalise_idft else "forward",
        )
