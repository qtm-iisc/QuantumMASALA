from __future__ import annotations

__all__ = ["GSpaceCuPy", "GkSpaceCuPy"]

from functools import lru_cache
import numpy as np
import cupy as cp

from qtm.lattice import ReciLattice
from qtm.gspace import GSpace, GkSpace

from qtm.fft import FFT3DFull

from qtm.config import NDArray


class FFT3DFullCuPy(FFT3DFull):
    """CuPy version of `qtm.gspace.fft.FFT3DFull`

    This class is required because CuPy arrays does not support keyword
    argument 'mode' in its `cp.ndarray.take` method.

    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        idxgrid: NDArray,
        normalise_idft: bool,
        backend: str | None = "cupy",
    ):
        super().__init__(shape, idxgrid, normalise_idft, "cupy")


class GSpaceCuPy(GSpace):
    FFT3D = FFT3DFullCuPy
    _normalise_idft = True

    def __init__(
        self,
        recilat: ReciLattice,
        ecut: float,
        grid_shape: tuple[int, int, int] | None = None,
    ):
        assert isinstance(recilat, ReciLattice)
        assert isinstance(recilat.recvec, cp.ndarray)
        super().__init__(recilat, ecut, grid_shape)

    @classmethod
    @lru_cache(maxsize=None)
    def from_cpu(cls, gspc_cpu: GSpace):
        assert isinstance(gspc_cpu, GSpace)
        assert isinstance(gspc_cpu.g_cryst, np.ndarray)

        recilat_cpu = gspc_cpu.recilat
        recilat = ReciLattice(recilat_cpu.tpiba, cp.asarray(recilat_cpu.recvec))

        ecut = gspc_cpu.ecut
        grid_shape = gspc_cpu.grid_shape

        return cls(recilat, ecut, grid_shape)


class GkSpaceCuPy(GkSpace):
    FFT3D = FFT3DFullCuPy
    _normalise_idft = False

    def __init__(self, gspc: GSpaceCuPy, k_cryst: tuple[float, float, float]):
        if not isinstance(gspc, GSpaceCuPy):
            raise TypeError(
                f"'gspc' must be a '{GSpaceCuPy}' instance. " f"got '{type(gspc)}'"
            )
        super().__init__(gspc, k_cryst)

    @classmethod
    @lru_cache(maxsize=None)
    def from_cpu(cls, gkspc_cpu: GkSpace):
        assert isinstance(gkspc_cpu, GkSpace)
        assert isinstance(gkspc_cpu.gk_cryst, np.ndarray)

        return cls(GSpaceCuPy.from_cpu(gkspc_cpu.gwfn), gkspc_cpu.k_cryst)
