from __future__ import annotations
__all__ = ['Wavefun', 'WavefunG', 'WavefunR',
           'WavefunSpinG', 'WavefunSpinR']

from abc import ABC, abstractmethod
import numpy as np

from qtm.gspace import GkSpace
from .buffer import Buffer
from .field import FieldR
from .gemm_wrappers import get_zgemm
from qtm import qtmconfig

from qtm.config import NDArray
from qtm.msg_format import *


class Wavefun(Buffer, ABC):
    """Represents the (periodic part of) Bloch Wavefunctions of crystal.

    The G-Space of the bloch wavefunctions are centred at a given k-point
    instead of the origin. This G-Space is described by `qtm.gspace.GkSpace`
    instances. For naming consistency, this class will have the `gkspc`
    attribute aliasing `gspc`, which is a `qtm.gspace.GkSpace` instance.
    """
    numspin: int = None
    gspc: GkSpace

    @abstractmethod
    def __new__(cls, gkspc: GkSpace, data: NDArray):
        if cls.numspin is None:
            raise NotImplementedError(
                f"{Wavefun} subclass did not set class attribute 'numspin'."
            )
        return Buffer.__new__(cls, gkspc, data)

    def __init__(self, gkspc: GkSpace, data: NDArray):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(type_mismatch_msg('gspc', gkspc, GkSpace))
        Buffer.__init__(self, gkspc, data)
        self.gkspc: GkSpace = self.gspc
        """Alias of `gspc`"""
        self._zgemm = get_zgemm(type(data))
        """Wrapper to ZGEMM BLAS routine. Refer to 
        `qtm.containers.gemm_wrappers.ZGEMMWrapper` for further info"""


class WavefunG(Wavefun):
    numspin: int = 1

    def __new__(cls, gkspc: GkSpace, data: NDArray):
        if not qtmconfig.mpi4py_installed:
            return Wavefun.__new__(cls, gkspc, data)

        from qtm.mpi import DistGkSpace, DistWavefun, DistWavefunG
        if isinstance(gkspc, DistGkSpace):
            return DistWavefun.__new__(DistWavefunG, gkspc, data)
        return Wavefun.__new__(cls, gkspc, data)

    @classmethod
    def _get_basis_size(cls, gkspc: GkSpace):
        return cls.numspin * gkspc.size_g

    def __init__(self, gkspc: GkSpace, data: NDArray):
        Wavefun.__init__(self, gkspc, data)

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

    def norm2(self):
        vdot = self.gspc.allocate_array(self.shape)
        for iwfn, wfn in enumerate(self.data.reshape((-1, self.basis_size))):
            vdot.ravel()[iwfn] = np.vdot(wfn, wfn)

        if hasattr(self.gkspc, 'pwgrp_comm'):
            comm = self.gkspc.pwgrp_comm
            comm.Allreduce(comm.IN_PLACE, vdot, comm.SUM)
        return vdot

    def norm(self):
        return np.sqrt(self.norm2())

    def normalize(self):
        self.data[:] /= self.norm()[:, np.newaxis]

    def vdot(self, ket: WavefunG):
        # TODO: Reprofile this and optimize it for memory usage
        # TODO: Reevaluate this implementation when SciPy is compiled with CBLAS too
        # TODO: Add check for C-Contiguity
        if not isinstance(ket, type(self)):
            raise TypeError(type_mismatch_msg('ket', ket, type(self)))
        if self.gkspc is not ket.gkspc:
            raise ValueError(obj_mismatch_msg(
                'self.gkspc', self.gkspc, 'ket.gkspc', ket.gkspc
            ))

        bra_ = self.data.reshape((-1, self.basis_size))
        ket_ = ket.data.reshape((-1, self.basis_size))
        braket = self._zgemm(
            alpha=1.0, a=bra_.T, trans_a=2,
            b=ket_.T, trans_b=0,
        )
        # braket = bra_.conj() @ ket_.T

        if self.shape == ():
            return braket.reshape(ket.shape)
        elif ket.shape == ():
            return braket.reshape(self.shape)
        else:
            return braket.reshape((*self.shape, *ket.shape))


class WavefunR(Wavefun):
    numspin: int = 1

    def __new__(cls, gkspc: GkSpace, data: NDArray):
        if not qtmconfig.mpi4py_installed:
            return Wavefun.__new__(cls, gkspc, data)

        from qtm.mpi import DistGkSpace, DistWavefun, DistWavefunR
        if isinstance(gkspc, DistGkSpace):
            return DistWavefun.__new__(DistWavefunR, gkspc, data)
        return Wavefun.__new__(cls, gkspc, data)

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

    def get_density(self, normalize: bool = True) -> FieldR:
        gwfn = self.gkspc.gwfn
        den_data = self.data.conj()
        den_data *= self.data
        den_data = den_data.reshape((*self.shape, self.numspin, -1))
        den = FieldR(gwfn, den_data)
        if not normalize:
            return den
        fac = np.sum(den, axis=-1) * gwfn.reallat_dv
        den /= fac[:, np.newaxis]
        return den


class WavefunSpinG(WavefunG):
    numspin = 2

    def __new__(cls, gkspc: GkSpace, data: NDArray):
        if not qtmconfig.mpi4py_installed:
            return Wavefun.__new__(cls, gkspc, data)

        from qtm.mpi import DistGkSpace, DistWavefun, DistWavefunSpinG
        if isinstance(gkspc, DistGkSpace):
            return DistWavefun.__new__(DistWavefunSpinG, gkspc, data)
        return Wavefun.__new__(cls, gkspc, data)

    def to_r(self):
        gkspc = self.gkspc
        data_g = self.data.reshape((*self.shape, self.numspin, -1))

        data_r = gkspc.g2r(data_g).reshape((*self.shape, -1))
        return WavefunSpinR(gkspc, data_r)


class WavefunSpinR(WavefunR):
    numspin = 2

    def __new__(cls, gkspc: GkSpace, data: NDArray):
        if not qtmconfig.mpi4py_installed:
            return Wavefun.__new__(cls, gkspc, data)

        from qtm.mpi import DistGkSpace, DistWavefun, DistWavefunR
        if isinstance(gkspc, DistGkSpace):
            return DistWavefun.__new__(DistWavefunR, gkspc, data)
        return Wavefun.__new__(cls, gkspc, data)

    def to_g(self):
        gkspc = self.gkspc
        data_r = self.data.reshape((*self.shape, self.numspin, -1))

        data_g = gkspc.r2g(data_r).reshape((*self.shape, -1))
        return WavefunSpinR(gkspc, data_g)
