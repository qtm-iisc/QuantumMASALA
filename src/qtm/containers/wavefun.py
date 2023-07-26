# from __future__ import annotations
from typing import Self
from qtm.config import NDArray
__all__ = ['WavefunG', 'WavefunR',
           'WavefunSpinG', 'WavefunSpinR']

from abc import ABC
from scipy.linalg.blas import zgemm

from qtm.gspace import GSpace, GkSpace
from .buffer import Buffer


class Wavefun(Buffer, ABC):

    numspin: int = None
    gspc: GkSpace

    def __init__(self, gkspc: GkSpace, data: NDArray):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(f"'gkspc' must be a '{GkSpace}' instance. "
                            f"got type {type(gkspc)}")
        Buffer.__init__(self, gkspc, data)
        self.gkspc: GkSpace = self.gspc
        self.gspc: GSpace = self.gkspc.gwfn


class WavefunG(Wavefun):
    numspin: int = 1

    @classmethod
    def _get_basis_size(cls, gkspc: GkSpace):
        return cls.numspin * gkspc.size_g

    @property
    def basis_type(self):
        return 'g'

    def to_g(self):
        return self

    def to_r(self):
        gkspc = self.gkspc
        data_g = self.data.reshape((*self.shape, self.numspin, -1))

        data_r = gkspc.g2r(data_g).reshape((*self.shape, -1))
        return WavefunR(gkspc, data_r)

    def vdot(self, ket: Self):
        # TODO: Reprofile this and optimize it for memory usage
        # TODO: Reevaluate this implementation when SciPy is compiled with CBLAS too
        if not isinstance(ket, type(self)):
            raise TypeError(f"'ket' must be a '{type(self)}' instance. "
                            f"got type '{type(ket)}'.")
        if self.gkspc != ket.gkspc:
            raise ValueError("mismatch in 'gkspc' between the two 'WavefunG' instnaces")

        # The transposed vies of bra and ket are generated so that the input args
        # are F-contiguous and the FBLAS wrapper need not make array copies.
        bra_ = self.data.reshape((-1, self.basis_size)).T
        ket_ = ket.data.reshape((-1, self.basis_size)).T
        braket = self.gkspc.create_buffer((bra_.shape[1], ket_.shape[1]))

        # The transposed vies of bra and ket are passed so that the input args
        # are F-contiguous and the FBLAS wrapper need not make array copies.
        braket[:] = zgemm(
            alpha=1.0, a=bra_, trans_a=2,
            b=ket_, trans_b=0,
        )
        return braket.reshape((*self.shape, *ket.shape))


class WavefunR(Wavefun):
    numspin: int = 1

    @classmethod
    def _get_basis_size(cls, gkspc: GkSpace):
        return cls.numspin * gkspc.size_r

    @property
    def basis_type(self):
        return 'r'

    def to_r(self):
        return self

    def to_g(self):
        gkspc = self.gkspc
        data_r = self.data.reshape((*self.shape, self.numspin, -1))

        data_g = gkspc.r2g(data_r).reshape((*self.shape, -1))
        return WavefunG(gkspc, data_g)


class WavefunSpinG(WavefunG):
    numspin = 2

    def to_r(self):
        gkspc = self.gkspc
        data_g = self.data.reshape((*self.shape, self.numspin, -1))

        data_r = gkspc.g2r(data_g).reshape((*self.shape, -1))
        return WavefunSpinR(gkspc, data_r)


class WavefunSpinR(WavefunR):
    numspin = 2

    def to_g(self):
        gkspc = self.gkspc
        data_r = self.data.reshape((*self.shape, self.numspin, -1))

        data_g = gkspc.r2g(data_r).reshape((*self.shape, -1))
        return WavefunSpinR(gkspc, data_g)
