__all__ = ['TaylorExp']
from copy import deepcopy
import numpy as np
from qtm.containers.field import FieldRType
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.logger import qtmlogger
from qtm.pseudo.nloc import NonlocGenerator
from qtm.tddft_gamma.expoper.base import TDExpOperBase
from scipy.linalg.blas import zaxpy


class TaylorExp(TDExpOperBase):

    __slots__ = ['order']

    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: FieldRType, l_nloc: list[NonlocGenerator],
                 time_step: float, order: int = 4):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc,
                         time_step)

        if not isinstance(order, int) or order < 1:
            raise ValueError("'order' must be a positive integer. "
                             f"got '{order}' (type {type(order)})")
        self.order = order

    def prop_psi(self, l_psi: list[KSWfn], l_prop_psi: list[KSWfn]):
        # l_prop_psi[:] = l_psi
        if self.is_noncolin:
            qtmlogger.warning("TaylorExp.prop_psi(): is_noncolin not implemented yet.")
            # l_psi = l_psi.reshape(
            #     (1, -1, self.gkspc.numgk * (1 + self.is_noncolin))
            # )
            # l_prop_psi = l_prop_psi.reshape(
            #     (1, -1, self.gkspc.numgk * (1 + self.is_noncolin))
            # )

        for idxspin in range(1 + self.is_spin * (not self.is_noncolin)):
            if self.is_spin:
                self.set_idxspin(idxspin)
            psi, prop_psi = deepcopy(l_psi[idxspin].evc_gk), l_prop_psi[idxspin].evc_gk
            hpsi = psi.copy()
            hpsi*=0.0

            fac = 1
            for iorder in range(self.order):
                self.h_psi(psi, hpsi)
                fac *= -1j * self.time_step / (iorder + 1)
                # prop_psi += fac * hpsi
                zaxpy(x=hpsi._data.reshape(-1), y=prop_psi._data.reshape(-1), a=fac)
                psi, hpsi = hpsi, psi
