from typing import Union
from qtm.config import NDArray

import numpy as np

from qtm.gspace import GkSpace
from qtm.containers import FieldR, WavefunG, WavefunSpinG


class KSWavefun:

    def __init__(self, gkspc: GkSpace, numbnd: int, is_noncolin: bool):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(f"'gkspc' must be a '{GkSpace}' instance. "
                            f"got '{type(gkspc)}'.")
        self.gkspc: GkSpace = gkspc
        self.k_cryst: tuple[float, float, float] = self.gkspc.k_cryst

        if not isinstance(numbnd, int) or numbnd <= 0:
            raise ValueError(f"'numbnd' must be a positive integer. "
                             f"got {numbnd} (type {type(numbnd)}).")
        self.numbnd: int = numbnd

        if not isinstance(is_noncolin, bool):
            raise TypeError(f"'is_noncolin' must be a boolean. "
                            f"got '{type(is_noncolin)}. ")
        self.is_noncolin: bool = is_noncolin

        if self.is_noncolin:
            evc = WavefunSpinG.empty(self.gkspc, numbnd)
        else:
            evc = WavefunG.empty(self.gkspc, numbnd)
        self.evc: Union[WavefunG, WavefunSpinG] = evc

        self.evl: NDArray = self.gkspc.create_buffer(self.numbnd)
        # self.occ: NDArray = self.gkspc.create_buffer(self.numbnd)
        
        self.occ: np.ndarray = np.empty((1, self.numbnd), dtype='f8')
        """(``(1+self.is_spin, self.numbnd)``, ``'f8'``) List of occupation numbers
        """

    def normalize(self):
        self.evc /= self.evc.norm()

    def compute_rho(self):
        numspin = 1 + self.is_noncolin
        rho_r = FieldR.zeros(self.gkspc.gwfn, numspin)

        for ibnd in range(self.numbnd):
            wfn = self.evc[ibnd]
            rho_r.r[:] += self.evl[ibnd] * \
                np.abs(wfn.to_r().r ** 2).reshape((numspin, -1))

        return rho_r
