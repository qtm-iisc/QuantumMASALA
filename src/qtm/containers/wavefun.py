# from __future__ import annotations
from qtm.config import NDArray
__all__ = ['WavefunG', 'WavefunR',
           'WavefunSpinG', 'WavefunSpinR']

from abc import abstractmethod
from .buffer import Buffer
from qtm.gspace import GkSpace


class Wavefun(Buffer):

    numspin: int = None
    gspc: GkSpace

    @classmethod
    def _get_basis_size(cls, gkspc: GkSpace) -> int:
        pass

    def __init__(self, gkspc: GkSpace, data: NDArray):
        if not isinstance(gkspc, GkSpace):
            raise TypeError("'gkspc' must be a GkSpace instance. "
                            f"got type {type(gkspc)}")
        super().__init__(gkspc, data)
        self.gkspc: GkSpace = self.gspc

    @property
    @abstractmethod
    def basis_type(self) -> str:
        pass

    @abstractmethod
    def to_rwavefun(self):
        pass

    @abstractmethod
    def to_gwavefun(self):
        pass


class WavefunG(Wavefun):

    numspin: int = 1

    @classmethod
    def _get_basis_size(cls, gkspc: GkSpace):
        return cls.numspin * gkspc.size_g

    @property
    def basis_type(self):
        return 'g'

    def to_gwavefun(self):
        return self

    def to_rwavefun(self):
        gkspc = self.gkspc
        shape = self.shape[:-1]

        data_g = self.data.reshape((*shape, self.numspin, -1))
        data_r = gkspc.r2g(data_g).reshape((*shape, -1))
        return WavefunR(gkspc, data_r)


class WavefunR(Wavefun):

    numspin: int = 1

    @classmethod
    def _get_basis_size(cls, gkspc: GkSpace):
        return cls.numspin * gkspc.size_r

    @property
    def basis_type(self):
        return 'r'

    def to_rwavefun(self):
        return self

    def to_gwavefun(self):
        gkspc = self.gkspc
        shape = self.shape[:-1]

        data_r = self.data.reshape((*shape, self.numspin, -1))
        data_g = gkspc.r2g(data_r).reshape((*shape, -1))
        return WavefunG(gkspc, data_g)


class WavefunSpinG(WavefunG):
    numspin = 2


class WavefunSpinR(WavefunR):
    numspin = 2
