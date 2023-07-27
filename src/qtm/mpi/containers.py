# from __future__ import annotations
from typing import Self
from qtm.config import NDArray

from abc import ABC
import numpy as np
from mpi4py import MPI

from qtm.containers import (
    Buffer, FieldR, FieldG,
    Wavefun, WavefunG, WavefunR,
    WavefunSpinG, WavefunSpinR
)
from .gspace import DistGSpaceBase, DistGkSpace


class DistBuffer(Buffer, ABC):

    gspc: DistGSpaceBase

    _mpi_op_map = {
        np.add: MPI.SUM, np.prod: MPI.PROD,
        np.maximum: MPI.MAX, np.minimum: MPI.MIN,  # NOTE: Propagates NaN's; np.fmax and np.fmin doesn't
        np.logical_and: MPI.LAND, np.bitwise_and: MPI.BAND,
        np.logical_or: MPI.LOR, np.bitwise_or: MPI.BOR,
        np.logical_xor: MPI.LXOR, np.bitwise_xor: MPI.BXOR
    }

    def __init__(self, dist_gspc: DistGSpaceBase, data: NDArray):
        if not isinstance(dist_gspc, DistGSpaceBase):
            raise TypeError(f"'dist_gspc' must be a '{DistGSpaceBase}' instance. "
                            f"got '{type(dist_gspc)}'")
        Buffer.__init__(self, dist_gspc, data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = Buffer.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)

        if isinstance(out, Buffer):
            return out

        # If not a buffer instance, it implies the length of the last dimension
        # has changed .As only ufuncs allowed are scalar and reduction,
        # this leaves only reduction, hence a collective reduction is called
        if ufunc not in DistBuffer._mpi_op_map:
            # WARNING: assuming dtype is 'c16' as buffer arrays must be 'c16' and
            # thus all operands will be typecast to 'c16'
            def new_op(inpbuf, outbuf, dtype):
                inpbuf = np.frombuffer(inpbuf, dtype='c16')
                outbuf = np.frombuffer(outbuf, dtype='c16')
                ufunc(inpbuf, outbuf, out=outbuf)
            mpi_op = MPI.Op.Create(new_op, False)
            self._mpi_op_map[ufunc] = mpi_op

        self.gspc.pwgrp_comm.Allreduce(MPI.IN_PLACE, out, self._mpi_op_map[ufunc])
        return out


class DistFieldG(DistBuffer, FieldG):

    def __init__(self, dist_gspc: DistGSpaceBase, data: NDArray):
        DistBuffer.__init__(self, dist_gspc, data)

    def to_r(self):
        gspc = self.gspc
        data = gspc.g2r(self.data)
        return DistFieldR(gspc, data)


class DistFieldR(DistBuffer, FieldR):

    def __init__(self, dist_gspc: DistGSpaceBase, data: NDArray):
        DistBuffer.__init__(self, dist_gspc, data)

    def to_g(self):
        gspc = self.gspc
        data = gspc.r2g(self.data)
        return DistFieldG(gspc, data)


class DistWavefun(DistBuffer, Wavefun, ABC):

    gspc: DistGkSpace

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        if not isinstance(dist_gkspc, DistGkSpace):
            raise TypeError(f"'gkspc' must be a '{DistGkSpace}' instance. "
                            f"got type {type(dist_gkspc)}")
        DistBuffer.__init__(self, dist_gkspc, data)
        self.gkspc = self.gspc


class DistWavefunG(DistWavefun, WavefunG):

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        DistWavefun.__init__(self, dist_gkspc, data)
        self._norm_fac = float(
            np.sqrt(self.gkspc.reallat_cellvol) / np.prod(self.gspc.grid_shape)
        )

    def to_r(self):
        gkspc = self.gkspc
        data_g = self.data.reshape((*self.shape, self.numspin, -1))

        data_r = gkspc.g2r(data_g).reshape((*self.shape, -1))
        return DistWavefunR(gkspc, data_r)

    def vdot(self, ket: Self):
        out = WavefunG.vdot(self, ket)
        comm = self.gkspc.pwgrp_comm
        comm.Allreduce(comm.IN_PLACE, out, comm.SUM)
        return out


class DistWavefunR(DistWavefun, WavefunR):

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        DistWavefun.__init__(self, dist_gkspc, data)

    def to_g(self):
        gkspc = self.gkspc
        data_r = self.data.reshape((*self.shape, self.numspin, -1))

        data_g = gkspc.r2g(data_r).reshape((*self.shape, -1))
        return WavefunG(gkspc, data_g)


class DistWavefunSpinG(DistWavefunG):
    numspin = 2

    def to_r(self):
        gkspc = self.gkspc
        data_g = self.data.reshape((*self.shape, self.numspin, -1))

        data_r = gkspc.g2r(data_g).reshape((*self.shape, -1))
        return DistWavefunSpinR(gkspc, data_r)


class DistWavefunSpinR(DistWavefunR):
    numspin = 2

    def to_g(self):
        gkspc = self.gkspc
        data_r = self.data.reshape((*self.shape, self.numspin, -1))

        data_g = gkspc.r2g(data_r).reshape((*self.shape, -1))
        return DistWavefunSpinR(gkspc, data_g)
