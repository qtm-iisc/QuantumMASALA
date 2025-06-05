from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
__all__ = [
    "BufferType",
    "DistBufferType",
    "FieldGType",
    "get_FieldG",
    "get_DistFieldG",
    "FieldRType",
    "get_FieldR",
    "get_DistFieldR",
    "WavefunGType",
    "get_WavefunG",
    "get_DistWavefunG",
    "WavefunRType",
    "get_WavefunR",
    "get_DistWavefunR",
]
from abc import ABC
from functools import lru_cache
import numpy as np
from mpi4py import MPI

from qtm.containers import (
    BufferType,
    FieldGType,
    get_FieldG,
    FieldRType,
    get_FieldR,
    WavefunGType,
    get_WavefunG,
    WavefunRType,
    get_WavefunR,
    WavefunType,
)
from .gspace import DistGSpaceBase, DistGSpace, DistGkSpace

from qtm.msg_format import type_mismatch_msg
from qtm.config import NDArray


class DistBufferType(BufferType, ABC):
    gspc: DistGSpaceBase
    BufferType: type[BufferType]

    _mpi_op_map = {
        np.add: MPI.SUM,
        np.prod: MPI.PROD,
        np.maximum: MPI.MAX,
        np.minimum: MPI.MIN,  # NOTE: Propagates NaN's; np.fmax and np.fmin doesn't
        np.logical_and: MPI.LAND,
        np.bitwise_and: MPI.BAND,
        np.logical_or: MPI.LOR,
        np.bitwise_or: MPI.BOR,
        np.logical_xor: MPI.LXOR,
        np.bitwise_xor: MPI.BXOR,
    }

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = BufferType.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)

        if isinstance(out, BufferType):
            return out

        # If not a buffer instance, it implies the length of the last dimension
        # has changed .As only ufuncs allowed are scalar and reduction,
        # this leaves only reduction, hence a collective reduction is called
        if ufunc not in DistBufferType._mpi_op_map:
            # WARNING: assuming dtype is 'c16' as buffer arrays must be 'c16' and
            # thus all operands will be typecast to 'c16'
            def new_op(inpbuf, outbuf, dtype):
                inpbuf = np.frombuffer(inpbuf, dtype="c16")
                outbuf = np.frombuffer(outbuf, dtype="c16")
                ufunc(inpbuf, outbuf, out=outbuf)

            mpi_op = MPI.Op.Create(new_op, False)
            DistBufferType._mpi_op_map[ufunc] = mpi_op

        # mpi4py does not treat NumPy scalars as buffer-like, so they are
        # reshaped to single-element arrays before comm and back after
        is_scalar = False
        if out.shape == ():
            is_scalar = True
            out = out.reshape((-1,))
        self.gspc.pwgrp_comm.Allreduce(MPI.IN_PLACE, out, self._mpi_op_map[ufunc])
        return out if not is_scalar else out[0]

    def gather(self, allgather: bool) -> Self | None:
        if not isinstance(allgather, bool):
            raise TypeError(type_mismatch_msg("allgather", allgather, bool))

        buftype = self.BufferType
        # FIXME: change the try-except block to if-else block
        # if issubclass(buftype, WavefunType):
        try:
            data = self.data.reshape((*self.shape, self.numspin, -1))
        except:
            data = self.data
        if self.basis_type == "r":
            data_glob = self.gspc.gather_r(data, allgather)
        else:  # if self.basis_type == 'g'
            data_glob = self.gspc.gather_g(data, allgather)

        gspc_glob = self.gspc.gspc_glob
        if allgather or self.gspc.pwgrp_rank == 0:
            return buftype(data_glob.reshape((*self.shape, -1)))

    def allgather(self) -> BufferType:
        return self.gather(allgather=True)

    @classmethod
    def scatter(cls, buf_glob: BufferType) -> DistBufferType:
        with cls.gspc.pwgrp_comm as comm:
            is_root = comm.rank == 0
            if is_root:
                if type(buf_glob) is not cls.BufferType:
                    raise TypeError()
                if cls.gspc.gspc_glob is not buf_glob.gspc:
                    raise ValueError()

            shape = comm.bcast(buf_glob.shape if is_root else None)
            basis_type = comm.bcast(buf_glob.basis_type if is_root else None)
            data_glob = buf_glob.data if is_root else None
            try:
                if is_root:  # and issubclass(cls.BufferType, WavefunType):
                    data_glob = data_glob.reshape(
                        (*buf_glob.shape, buf_glob.numspin, -1)
                    )
            except:
                pass

            if basis_type == "g":
                data_loc = cls.gspc.scatter_g(data_glob)
            else:  # if basis_type == 'r':
                data_loc = cls.gspc.scatter_r(data_glob)
            return cls(data_loc.reshape((*shape, -1)))


@lru_cache(maxsize=None)
def get_DistFieldG(dist_gspc: DistGSpace) -> type[FieldGType]:
    if not isinstance(dist_gspc, DistGSpace):
        raise TypeError(type_mismatch_msg("dist_gspc", dist_gspc, DistGSpace))

    class DistFieldG(DistBufferType, FieldGType, gspc=dist_gspc):
        gspc: DistGSpace
        BufferType = get_FieldG(dist_gspc.gspc_glob)

        @property
        def data_g0(self) -> NDArray:
            with self.gspc.pwgrp_comm as comm:
                return comm.bcast(self.data[..., 0], root=0)

    return DistFieldG


@lru_cache(maxsize=None)
def get_DistFieldR(dist_gspc: DistGSpace) -> type[FieldRType]:
    if not isinstance(dist_gspc, DistGSpace):
        raise TypeError(type_mismatch_msg("dist_gspc", dist_gspc, DistGSpace))
    class DistFieldR(DistBufferType, FieldRType, gspc=dist_gspc):
        gspc: DistGSpace
        BufferType = get_FieldR(dist_gspc.gspc_glob)
    return DistFieldR

@lru_cache(maxsize=None)
def get_DistWavefunG(dist_gkspc: DistGkSpace, numspin: int) -> type[WavefunGType]:
    if not isinstance(dist_gkspc, DistGkSpace):
        raise TypeError(type_mismatch_msg("dist_gspc", dist_gkspc, DistGkSpace))
    class DistWavefunG(DistBufferType, WavefunGType, gkspc=dist_gkspc, numspin=numspin):
        gspc: DistGkSpace
        gkspc: DistGkSpace
        BufferType = get_WavefunG(dist_gkspc.gkspc_glob, numspin)
        def vdot(self, ket: WavefunGType) -> NDArray:
            out = super().vdot(ket)
            comm = self.gkspc.pwgrp_comm
            comm.Allreduce(comm.IN_PLACE, out, comm.SUM)
            return out
    return DistWavefunG


@lru_cache(maxsize=None)
def get_DistWavefunR(dist_gkspc: DistGkSpace, numspin: int) -> type[WavefunRType]:
    if not isinstance(dist_gkspc, DistGkSpace):
        raise TypeError(type_mismatch_msg("dist_gspc", dist_gkspc, DistGkSpace))

    class DistWavefunR(DistBufferType, WavefunRType, gkspc=dist_gkspc, numspin=numspin):
        gspc: DistGkSpace
        gkspc: DistGkSpace
        BufferType = get_WavefunR(dist_gkspc.gkspc_glob, numspin)

    return DistWavefunR
