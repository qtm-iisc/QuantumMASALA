# from __future__ import annotations
__all__ = ['Field', 'FieldG', 'FieldR']

from abc import ABC, abstractmethod
from .buffer import Buffer
from qtm.gspace import GSpaceBase


class Field(Buffer, ABC):

    @classmethod
    @abstractmethod
    def _get_basis_size(cls, gspc: GSpaceBase) -> int:
        pass

    @property
    @abstractmethod
    def basis_type(self) -> str:
        pass

    @abstractmethod
    def to_fieldg(self):
        pass

    @abstractmethod
    def to_fieldr(self):
        pass


class FieldG(Buffer):

    @classmethod
    def _get_basis_size(cls, gspc: GSpaceBase):
        return gspc.size_g

    @property
    def basis_type(self):
        return 'g'

    def to_fieldr(self):
        gspc = self.gspc
        data = gspc.g2r(self.data)
        return FieldR(gspc, data)

    def to_fieldg(self):
        return self


class FieldR(Field):

    @classmethod
    def _get_basis_size(cls, gspc: GSpaceBase):
        return gspc.size_r

    @property
    def basis_type(self):
        return 'r'

    def to_fieldr(self):
        return self

    def to_fieldg(self):
        gspc = self.gspc
        data = gspc.r2g(self.data)
        return FieldG(gspc, data)
