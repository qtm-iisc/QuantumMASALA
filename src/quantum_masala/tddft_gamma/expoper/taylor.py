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

    def prop_psi(self, l_psi: np.ndarray, l_prop_psi: np.ndarray):
        l_prop_psi[:] = l_psi
        if self.is_noncolin:
            l_psi = l_psi.reshape(
                (1, -1, self.gkspc.numgk * (1 + self.is_noncolin))
            )
            l_prop_psi = l_prop_psi.reshape(
                (1, -1, self.gkspc.numgk * (1 + self.is_noncolin))
            )

        for idxspin in range(1 + self.is_spin * (not self.is_noncolin)):
            if self.is_spin:
                self.set_idxspin(idxspin)
            psi, prop_psi = np.copy(l_psi[idxspin]), l_prop_psi[idxspin]
            hpsi = np.empty_like(psi)

            fac = 1
            for iorder in range(self.order):
                self.h_psi(psi, hpsi)
                fac *= -1j * self.time_step / (iorder + 1)
                prop_psi += fac * hpsi
                psi, hpsi = hpsi, psi
