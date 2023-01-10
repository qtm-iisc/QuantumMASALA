__all__ = ['TaylorExp']
import numpy as np

from quantum_masala.core import GkSpace, RField
from quantum_masala.pseudo import NonlocGenerator

from .base import TDExpOperBase


class TaylorExp(TDExpOperBase):

    __slots__ = ['order']

    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: RField, l_nloc: list[NonlocGenerator],
                 time_step: float, order: int = 4):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc,
                         time_step)

        if not isinstance(order, int) or order < 1:
            raise ValueError("'order' must be a positive integer. "
                             f"got '{order}' (type {type(order)})")
        self.order = order

    def prop_psi(self, l_psi_: np.ndarray, l_prop_psi_: np.ndarray):
        l_prop_psi_[:] = l_psi_
        if self.is_noncolin:
            l_psi_ = l_psi_.reshape((1, -1, l_psi_.shape[-1]))
            l_prop_psi_ = l_prop_psi_.reshape((1, -1, l_prop_psi_.shape[-1]))

        numspin = 1 + (self.is_spin and not self.is_noncolin)
        psi, hpsi = np.copy(l_psi_[0]), np.empty_like(l_psi_[0])
        for ispin in range(numspin):
            self.idxspin = ispin
            l_psi, l_prop_psi = l_psi_[ispin], l_prop_psi_[ispin]
            fac = 1
            for iorder in range(self.order):
                self.h_psi(psi, hpsi)
                fac *= -1j * self.time_step / (iorder + 1)
                l_prop_psi += fac * hpsi
                psi, hpsi = hpsi, psi
