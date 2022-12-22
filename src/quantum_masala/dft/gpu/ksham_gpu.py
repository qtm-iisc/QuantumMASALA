from warnings import warn

import numpy as np
import cupy as cp

from quantum_masala.core import GkSpace, GField
from quantum_masala.pseudo import NonlocGenerator
from quantum_masala.dft import KSHam
from .fftmod_gpu import CpFFTSlab


class KSHamGPU(KSHam):
    __slots__ = ["gkspc", "fft_mod", "ke_gk", "l_vkb", "l_vkb_H", "dij",
                 "noncolin", "numspin", "vloc_r", "idxspin"]

    def __init__(self, gkspc: GkSpace, numspin: int, noncolin: bool,
                 vloc: GField,
                 l_nloc: list[NonlocGenerator]):
        super().__init__(gkspc, numspin, noncolin, vloc, l_nloc)

        self.ke_gk = cp.asarray(self.ke_gk)

        self.l_vkb = cp.asarray(self.l_vkb)
        self.l_vkb_H = cp.asarray(self.l_vkb_H)
        self.dij = cp.asarray(self.dij)

        self.vloc_r = cp.asarray(self.vloc_r)

        self.fft_mod = CpFFTSlab.from_gkspc(self.gkspc)