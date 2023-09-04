__all__ = ['MixModBase']

from abc import ABC, abstractmethod
import numpy as np

from qtm.gspace import GSpace
from qtm.containers import FieldGType
from qtm.pot.utils import check_rho
from qtm.dft import DFTCommMod

from qtm.constants import FPI


class MixModBase(ABC):

    @abstractmethod
    def __init__(self, dftcomm: DFTCommMod, rho: FieldGType, beta: float, mixdim: int):
        assert isinstance(dftcomm, DFTCommMod)
        self.dftcomm = dftcomm
        self.is_root_pwgrp = self.dftcomm.pwgrp_inter_image.rank == 0

        check_rho(rho)
        self.FieldG: type[FieldGType] = type(rho)
        self.grho: GSpace = self.FieldG.gspc
        self.numspin: int = rho.shape[0]

        assert isinstance(beta, float) and 0 < beta <= 1
        self.beta: float = beta
        assert isinstance(mixdim, int) and mixdim > 1
        self.mixdim: int = mixdim

    def _check_rho(self, rho_in: FieldGType, rho_out: FieldGType):
        assert type(rho_in) is self.FieldG
        assert type(rho_out) is self.FieldG
        assert rho_in.shape == (self.numspin, )
        assert rho_out.shape == (self.numspin,)

    def _dot(self, rho1_g: FieldGType, rho2_g: FieldGType) -> float:
        chden1 = sum(rho1_g)
        chden2 = sum(rho2_g)
        fac = (
            0.5 * FPI * self.grho.reallat_dv / np.prod(self.grho.grid_shape)
        )

        with np.errstate(divide='ignore'):
            dotvec = chden1.conj() * chden2 / self.grho.g_norm2
            if self.grho.has_g0:
                dotvec.data[..., 0] = 0
        dot = fac * np.sum(dotvec).real

        if self.numspin == 2:
            spden1 = rho1_g[0] - rho1_g[1]
            spden2 = rho2_g[0] - rho2_g[1]
            tpiba = self.grho.recilat.tpiba
            dot += fac * np.sum(spden1.conj() * spden2).real / tpiba**2
        return dot

    def compute_error(self, rho_in: FieldGType, rho_out: FieldGType) -> float:
        self._check_rho(rho_in, rho_out)
        res = rho_in - rho_out
        return self._dot(res, res)

    def mix(self, rho_in: FieldGType, rho_out: FieldGType) -> FieldGType:
        self._check_rho(rho_in, rho_out)
        if self.is_root_pwgrp:
            rho_next = self._mix(rho_in, rho_out)
        else:
            rho_next = self.FieldG.empty(self.numspin)
        self.dftcomm.pwgrp_inter_image.Bcast(rho_next.data)
        return rho_next

    @abstractmethod
    def _mix(self, rho_in: FieldGType, rho_out: FieldGType) -> FieldGType:
        pass
