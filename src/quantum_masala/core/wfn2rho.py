from typing import Type, Optional
import numpy as np

from .cryst import Crystal
from .fft import FFTGSpace
from .rho import Rho
from .wfn import WfnK

EPS5 = 1E-5
EPS10 = 1E-10


class Wfn2Rho(Rho):
    WfnK: Type[WfnK]

    def __init__(self, crystal: Crystal, WfnK_: Type[WfnK],
                 l_wfnk_kgrp: Optional[list[WfnK]] = None,
                 symm_flag: bool = True):
        self.WfnK = WfnK_

        grho = self.WfnK.gspc
        numspin = self.WfnK.numspin
        fft_rho = FFTGSpace(grho)

        super().__init__(crystal, grho, numspin, fft_rho, symm_flag=symm_flag)
        if l_wfnk_kgrp:
            self.update(l_wfnk_kgrp)

    def update(self, l_wfnk: list[WfnK]):
        if not all(isinstance(wfnk, type(l_wfnk[0])) for wfnk in l_wfnk[1:]):
            raise ValueError("'l_wfnk' must contain all elements of the same type, which can"
                             "be generated from 'WavefunK' routine")
        if not isinstance(l_wfnk[0], self.WfnK):
            raise ValueError("not compatible with 'l_wfnk'; input type does not match.\n"
                             f"Expected {self.WfnK}, got {type(l_wfnk[0])}")

        self._r[:] = 0
        for wfnk in l_wfnk:
            self._r[:] += wfnk.rho_r * wfnk.k_weight

        if self.pwcomm.is_kgrp_root:
            self.pwcomm.kgrp_intercomm.Allreduce_sum(self._r)

        self._g[:] = self.fft_rho.r2g(self._r)
        if self.symm_flag:
            self._g[:] = self._symmmod.symmetrize(self._g)
        self.sync()
        self._normalize()

        return self
