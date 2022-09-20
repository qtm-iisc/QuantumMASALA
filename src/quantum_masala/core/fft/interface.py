from typing import Optional

import numpy as np

from ..gspc import GSpace
from ..gspc_wfc import GSpaceWfc

from .config import FFT_METHOD


def FFTDriver_():
    """Returns the FFT Driver class based on `FFT_CONFIG` parameter"""
    if FFT_METHOD == "SLAB":
        from .driver import FFTDriverSlab as FFTDriver
    elif FFT_METHOD == "PENCIL":
        from .driver import FFTDriverPencil as FFTDriver
    else:
        raise ValueError(
            f"invalid value in 'FFT_CONFIG['METHOD']'. Must be one of the following:\n"
            f"'SLAB', 'PENCIL'. Got {FFT_METHOD}"
        )
    return FFTDriver


class FFTGSpace(FFTDriver_()):
    """FFT Interface for 'GSpace` instances.

    Functionally identical to `FFTDriver` class
    """

    def __init__(self, gspc: GSpace):
        super().__init__(gspc.grid_shape, gspc.idxgrid)


class FFTGSpaceWfc(FFTDriver_()):
    """FFT Interface for `GSpaceWfc` instances.
    """

    def __init__(self, gwfc: GSpaceWfc):
        grid_shape = gwfc.gspc.grid_shape
        super().__init__(grid_shape, gwfc.idxgrid)

    def gk2r(
        self, arr: np.ndarray, out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Method for performing Inverse FFT of input data in the truncated fourier space.

        Parameters
        ----------
        arr : np.ndarray
            Input List of fields represented in the truncated fourier space of k-point `idxk`.
        out : np.ndarray, optional
            Output array in which to place the result. Optional

        Returns
        -------
            Inverse FFT of data in Input.
        """

        return self.g2r(arr, out)

    def r2gk(self, arr_in: np.ndarray, out: Optional[np.ndarray] = None):
        """Method for performing FFT of input data in the real space.

        Parameters
        ----------
        arr_in : np.ndarray
            Input List of fields represented in the real space.
        out : np.ndarray, optional
            Output array in which to place the result. Optional

        Returns
        -------
            FFT of input fields, truncated to the space where
            :math:`\abs{\mathbf{G} + \mathbf{k}}^2 / 2 \leq E_cutwfc`
        """

        return self.r2g(arr_in, out)
