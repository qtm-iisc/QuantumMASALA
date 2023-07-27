# from __future__ import annotations
from typing import Union, Sequence
__all__ = ['FieldG', 'FieldR']

from abc import ABC, abstractmethod
import numpy as np

from qtm.gspace import GSpaceBase
from .buffer import Buffer


class FieldG(Buffer):

    @classmethod
    def _get_basis_size(cls, gspc: GSpaceBase):
        return gspc.size_g

    @property
    def basis_type(self):
        return 'g'

    def to_r(self):
        gspc = self.gspc
        data = gspc.g2r(self.data)
        return FieldR(gspc, data)

    def to_g(self):
        return self


class FieldR(Buffer):

    @classmethod
    def _get_basis_size(cls, gspc: GSpaceBase):
        return gspc.size_r

    @property
    def basis_type(self):
        return 'r'

    def to_r(self):
        return self

    def to_g(self):
        gspc = self.gspc
        data = gspc.r2g(self.data)
        return FieldG(gspc, data)

    def integrate_unitcell(self, axis: Union[int, Sequence[int]] = 0):
        return np.sum(
            np.sum(self, axis=-1),
            axis=axis
        ) * self.gspc.reallat_dv
