__all__ = ['TDExpOperBase', 'TDExpOper']

from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np

from quantum_masala.core import GkSpace, RField
from quantum_masala.pseudo import NonlocGenerator
from quantum_masala.dft import KSHam


class TDExpOperBase(KSHam, ABC):

    @abstractmethod
    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: RField, l_nloc: list[NonlocGenerator],
                 time_step: float):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc)
        self.time_step = time_step

    def update_vloc(self, vloc: RField):
        self._check_vloc(vloc)
        self.vloc_r = vloc.r

    def set_idxspin(self, idxspin: int):
        raise Exception("'set_idxspin' is not valid for 'TDExpOper'.")

    @abstractmethod
    def prop_psi(self, l_psi: np.ndarray, l_prop_psi: np.ndarray):
        pass


TDExpOper = TypeVar('TDExpOper', bound=TDExpOperBase)
