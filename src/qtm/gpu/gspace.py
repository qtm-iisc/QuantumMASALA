
import cupy as cp

from qtm.lattice import ReciLattice
from qtm.gspace import GSpace, GkSpace

from .fft import FFT3DFullCuPy


class GSpaceCuPy(GSpace):

    FFT3D = FFT3DFullCuPy

    def __init__(self, gspc_host: GSpace):
        recilat = gspc_host.recilat
        recilat = ReciLattice(recilat.tpiba, cp.asarray(recilat.recvec))

        ecut = gspc_host.ecut
        grid_shape = gspc_host.grid_shape

        super().__init__(recilat, ecut, grid_shape)


class GkSpaceCuPy(GkSpace):

    FFT3D = FFT3DFullCuPy

    def __init__(self, gspc_dev: GSpaceCuPy, k_cryst: tuple[float, float, float]):
        if not isinstance(gspc_dev, GSpaceCuPy):
            raise TypeError("'gspc' must be a 'GSpaceCuPy' instance. "
                            f"got {type(gspc_dev)}")
        super().__init__(gspc_dev, k_cryst)

