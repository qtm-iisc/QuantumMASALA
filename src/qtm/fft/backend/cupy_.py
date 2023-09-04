# TODO: Use CuFFT directly to implement in-place operations
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
__all__ = ['CuPyFFTWrapper']
import cupy as cp
from cupyx.scipy.fft import get_fft_plan, fftn, ifftn

from qtm.fft.backend.base import FFTBackend


class CuPyFFTWrapper(FFTBackend):
    """Wraps CuPy FFT routines in `cupyx.scipy.fft`"""

    ndarray = cp.ndarray

    def __init__(self, arr: tuple[int, ...],
                 axes: tuple[int, ...]):
        super().__init__(arr, axes)
        self.fft_plan = get_fft_plan(self._inp_fwd,
                                     axes=self.axes, value_type='C2C')

    @classmethod
    def allocate_array(cls, shape: int | Sequence[int],
                       dtype: str) -> cp.ndarray:
        return cp.empty(shape, dtype=dtype, order='C')

    def fft(self) -> cp.ndarray:
        return fftn(self._inp_fwd, axes=self.axes, plan=self.fft_plan,
                    overwrite_x=False)

    def ifft(self, normalise_idft: bool = False) -> cp.ndarray:
        return ifftn(self._inp_bwd, axes=self.axes, plan=self.fft_plan,
                     overwrite_x=True, norm='backward' if normalise_idft else 'forward')
