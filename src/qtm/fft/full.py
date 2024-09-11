from __future__ import annotations
__all__ = ['FFT3DFull', ]

from .base import FFT3D
from qtm.config import NDArray


class FFT3DFull(FFT3D):
    """3D FFT module that operates on all elements in the 3D grid, instead
    of skipping regions containing zeros.

    Refer to `qtm.fft.FFT3DSticks` regarding how the structure of G-Space
    can be used to reduce computation involved in FFT operations
    """
    def __init__(self, shape: tuple[int, int, int],
                 idxgrid: NDArray, normalise_idft: bool,
                 backend: str | None = None, **kwargs):
        super().__init__(shape, idxgrid, normalise_idft, backend, **kwargs)
        self.worker = self.FFTBackend(self.shape, (0, 1, 2),)
        self.worker.inp_bwd[:] = 0

    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self.worker.inp_fwd[:] = arr_inp
        out = self.worker.fft()
        out.take(self.idxgrid, out=arr_out)

    def g2r(self, arr_inp: NDArray, arr_out: NDArray=None) -> None:
        self.worker.inp_bwd.fill(0.0)
        self.worker.inp_bwd.put(self.idxgrid, arr_inp)
        if arr_out is not None:
            arr_out[:] = self.worker.ifft(self.normalise_idft)
        else:
            return self.worker.ifft(self.normalise_idft)
