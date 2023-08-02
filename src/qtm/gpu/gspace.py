from qtm.config import NDArray

import cupy as cp

from qtm.lattice import ReciLattice
from qtm.gspace import GSpace, GkSpace

from qtm.gspace.fft import FFT3DFull


class FFT3DFullCuPy(FFT3DFull):
    """CuPy version of `qtm.gspace.fft.FFT3DFull`

    This class is required because CuPy arrays does not support keyword
    argument 'mode' in its `cp.ndarray.take` method.

    """

    def __init__(self, shape: tuple[int, int, int],
                 idxgrid: NDArray, normalise_idft: bool
                 ):
        super().__init__(shape, idxgrid, normalise_idft, 'cupy')

    def r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self._work[:] = arr_inp
        self.worker.fft()
        self._work.take(self.idxgrid, out=arr_out)

    def g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        self._work[:].fill(0)
        self._work.put(self.idxgrid, arr_inp)
        self.worker.ifft(self.normalise_idft)
        arr_out[:] = self._work


class GSpaceCuPy(GSpace):

    FFT3D = FFT3DFullCuPy
    _normalise_idft = True

    def __init__(self, gspc_host: GSpace):
        recilat = gspc_host.recilat
        recilat = ReciLattice(recilat.tpiba, cp.asarray(recilat.recvec))

        ecut = gspc_host.ecut
        grid_shape = gspc_host.grid_shape

        super().__init__(recilat, ecut, grid_shape)


class GkSpaceCuPy(GkSpace):

    FFT3D = FFT3DFullCuPy
    _normalise_idft = False

    def __init__(self, gspc_dev: GSpaceCuPy, k_cryst: tuple[float, float, float]):
        if not isinstance(gspc_dev, GSpaceCuPy):
            raise TypeError(f"'gspc' must be a '{GSpaceCuPy}' instance. "
                            f"got '{type(gspc_dev)}'")
        super().__init__(gspc_dev, k_cryst)
