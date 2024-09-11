__all__ = ['TDExpOperBase', 'TDExpOper']

from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np
from qtm.containers.field import FieldRType
from qtm.dft.ksham import KSHam
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.pseudo.nloc import NonlocGenerator


class TDExpOperBase(KSHam, ABC):

    @abstractmethod
    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: FieldRType, l_nloc: list[NonlocGenerator],
                 time_step: float):
        super().__init__(gkspc, is_noncolin, vloc, l_nloc)
        self.time_step = time_step
        """Time step, in Hartree atomic units."""
        self.is_spin = is_spin

    def update_vloc(self, vloc: FieldRType):
        # FIXME: Uncomment later.
        # self._check_vloc(vloc)
        self.vloc = vloc.copy()

    def set_idxspin(self, idxspin: int):
        raise Exception("'set_idxspin' is not valid for 'TDExpOper'.")

    @abstractmethod
    def prop_psi(self, l_psi: KSWfn, l_prop_psi: KSWfn):
        pass


TDExpOper = TypeVar('TDExpOper', bound=TDExpOperBase)
