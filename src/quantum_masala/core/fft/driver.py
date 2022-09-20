from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .config import FFTLIB


def FFTLib_(*args):
    if FFTLIB == "NUMPY":
        from .lib import NpFFTLibWrapper as FFTLib
    if FFTLIB == "SCIPY":
        from .lib import SpFFTLibWrapper as FFTLib
    elif FFTLIB == "PYFFTW":
        from .lib import PyFFTWLibWrapper as FFTLib
    elif FFTLIB == "MKLFFT":
        from .lib import MKLFFTLibWrapper as FFTLib
    else:
        raise ValueError(
            "invalid value in 'FFT_LIB'. Must be one of the following:\n"
            "'NUMPY', 'SCIPY', 'PYFFTW', 'MKLFFT'.\n"
            f"Got {FFTLIB}"
        )
    return FFTLib(*args)


class FFTDriver(ABC):
    __slots__ = ["grid_shape", "idxgrid", "numgrid"]

    @abstractmethod
    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        idxgrid: tuple[list[int], list[int], list[int]],
    ):
        self.grid_shape = grid_shape
        self.idxgrid = idxgrid
        self.numgrid = len(self.idxgrid[0])

    def g2r(self, arr_in: np.ndarray, arr_out: Optional[np.ndarray] = None):
        arr_shape = arr_in.shape[:-1]
        arr_in = arr_in.reshape(-1, self.numgrid)
        n_arr = arr_in.shape[0]

        if arr_out is None:
            arr_out = np.empty((n_arr, *self.grid_shape), dtype="c16")
        arr_out = self._g2r(arr_in, arr_out)

        return arr_out.reshape(arr_shape + self.grid_shape)

    @abstractmethod
    def _g2r(self, arr_in: np.ndarray, arr_out: np.ndarray):
        pass

    def r2g(self, arr_in: np.ndarray, arr_out: Optional[np.ndarray] = None):
        arr_shape = arr_in.shape[:-3]
        arr_in = arr_in.reshape((-1, *self.grid_shape))
        n_arr = arr_in.shape[0]

        if arr_out is None:
            arr_out = np.empty((n_arr, self.numgrid), dtype="c16")
        arr_out = self._r2g(arr_in, arr_out)

        return arr_out.reshape((*arr_shape, self.numgrid))

    @abstractmethod
    def _r2g(self, arr_in: np.ndarray, arr_out: np.ndarray):
        pass


class FFTDriverSlab(FFTDriver):
    __slots__ = ["fft_slab"]

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        idxgrid: tuple[list[int], list[int], list[int]],
    ):
        super().__init__(grid_shape, idxgrid)

        self.fft_slab = FFTLib_(self.grid_shape, (0, 1, 2))

    def _g2r(self, arr_in: np.ndarray, arr_out: np.ndarray):
        n_arr = arr_in.shape[0]

        in_slab = np.zeros((n_arr, *self.grid_shape), dtype="c16")
        in_slab[(slice(None), *self.idxgrid)] = arr_in

        arr_out = self.fft_slab.do_ifft(in_slab, arr_out)
        return arr_out

    def _r2g(self, arr_in: np.ndarray, arr_out: np.ndarray):
        n_arr = arr_in.shape[0]

        out_slab = np.empty((n_arr, *self.grid_shape), dtype="c16")

        out_slab = self.fft_slab.do_fft(arr_in, out_slab)

        arr_out[:] = out_slab[(slice(None), *self.idxgrid)]
        return arr_out


class FFTDriverPencil(FFTDriver):
    __slots__ = ["nsticks", "isticks", "fft_sticks",
                 "islab", "fft_slab"]

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        idxgrid: tuple[list[int], list[int], list[int]],
    ):
        super().__init__(grid_shape, idxgrid)

        nx, ny, nz = self.grid_shape
        ix, iy, iz = self.idxgrid

        ixy = ix * ny + iy
        ixy_unique = np.unique(ixy)
        isearch = np.searchsorted(ixy_unique, ixy, side="left")
        self.nsticks = len(ixy_unique)

        self.isticks = (isearch, iz)

        self.fft_sticks = FFTLib_((self.nsticks, nz), 1)

        ix, iy = ixy_unique // ny, ixy_unique % ny

        self.islab = (ix, iy)

        self.fft_slab = FFTLib_((nx, ny, nz), (0, 1))

    def _g2r(self, arr_in: np.ndarray, arr_out: np.ndarray):
        n_arr = arr_in.shape[0]
        nx, ny, nz = self.grid_shape

        in_sticks = np.zeros((n_arr, self.nsticks, nz), dtype="c16")
        out_sticks = np.empty_like(in_sticks)
        in_sticks[(slice(None), *self.isticks)] = arr_in.reshape(-1, self.numgrid)
        out_sticks = self.fft_sticks.do_ifft(in_sticks, out_sticks)

        if arr_out is None:
            arr_out = np.empty((n_arr, nx, ny, nz), dtype="c16")
        in_slab = np.zeros((n_arr, nx, ny, nz), dtype="c16")
        out_slab = arr_out

        in_slab[(slice(None), *self.islab)] = out_sticks
        arr_out = self.fft_slab.do_ifft(in_slab, out_slab)

        return arr_out

    def _r2g(self, arr_in: np.ndarray, arr_out: np.ndarray):
        n_arr = arr_in.shape[0]
        nx, ny, nz = self.grid_shape

        in_slab = arr_in
        out_slab = np.empty((n_arr, nx, ny, nz), dtype="c16")

        out_slab = self.fft_slab.do_fft(in_slab, out_slab)

        in_sticks = np.zeros((n_arr, self.nsticks, nx), dtype="c16")
        out_sticks = np.empty((n_arr, self.nsticks, nx), dtype="c16")

        in_sticks[:] = out_slab[(slice(None), *self.islab)]
        out_sticks = self.fft_sticks.do_fft(in_sticks, out_sticks)
        arr_out[:] = out_sticks[(slice(None), *self.isticks)]

        return arr_out
