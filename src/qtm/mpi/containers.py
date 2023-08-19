from __future__ import annotations
__all__ = [
    'Buffer', 'Field', 'FieldG', 'FieldR', 'Wavefun', 'WavefunG', 'WavefunR',
    'WavefunSpinG', 'WavefunSpinR',
    'DistBuffer', 'DistField', 'DistFieldG', 'DistFieldR', 'DistWavefun',
    'DistWavefunG', 'DistWavefunR', 'DistWavefunSpinG', 'DistWavefunSpinR'
]
from abc import ABC
import numpy as np
from mpi4py import MPI

from qtm.containers import (
    Buffer, Field, FieldR, FieldG,
    Wavefun, WavefunG, WavefunR,
    WavefunSpinG, WavefunSpinR
)
from qtm.containers.gemm_wrappers import get_zgemm
from .gspace import DistGSpaceBase, DistGSpace, DistGkSpace

from qtm.config import NDArray
from qtm.msg_format import type_mismatch_msg


class DistBuffer(Buffer, ABC):

    gspc: DistGSpaceBase
    BufferType: type[Buffer]

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
            DistBuffer._mpi_op_map[ufunc] = mpi_op

        # mpi4py does not treat NumPy scalars as buffer-like, so they are
        # reshaped to single-element arrays before comm and back after
        is_scalar = False
        if out.shape == ():
            is_scalar = True
            out = out.reshape((-1, ))
        self.gspc.pwgrp_comm.Allreduce(MPI.IN_PLACE, out, self._mpi_op_map[ufunc])
        return out if not is_scalar else out[0]

    def gather(self, allgather: bool) -> Buffer | None:
        if not isinstance(allgather, bool):
            raise TypeError(type_mismatch_msg('allgather', allgather, bool))

        buftype = self.BufferType
        if issubclass(buftype, Wavefun):
            data = self.data.reshape((self.shape, self.numspin, -1))
        else:
            data = self.data

        if self.basis_type == 'r':
            data_glob = self.gspc.gather_r(data, allgather)
        else:  # if self.basis_type == 'g'
            data_glob = self.gspc.gather_g(data, allgather)

        gspc_glob = self.gspc.gspc_glob
        if allgather or self.gspc.pwgrp_rank == 0:
            return buftype(gspc_glob, data_glob.reshape((self.shape, -1)))

    def allgather(self) -> Buffer:
        return self.gather(allgather=True)

    @classmethod
    def scatter(cls, dist_gspc: DistGSpaceBase, buf_glob: Buffer) -> DistBuffer:
        if not isinstance(dist_gspc, DistGSpaceBase):
            raise TypeError(type_mismatch_msg(
                'dist_gspc', dist_gspc, DistGSpaceBase
            ))

        with dist_gspc.pwgrp_comm as comm:
            is_root = comm.rank == 0
            if is_root:
                if type(buf_glob) is not cls.BufferType:
                    raise TypeError()
                if dist_gspc.gspc_glob is not buf_glob.gspc:
                    raise ValueError()

            shape = comm.bcast(buf_glob.shape if is_root else None)
            basis_type = comm.bcast(buf_glob.basis_type if is_root else None)
            data_glob = buf_glob.data if is_root else None
            if is_root and issubclass(cls.BufferType, Wavefun):
                data_glob = data_glob.reshape(
                    (*buf_glob.shape, buf_glob.numspin, -1)
                )

            if basis_type == 'g':
                data_loc = dist_gspc.scatter_g(data_glob)
            else:  # if basis_type == 'r':
                data_loc = dist_gspc.scatter_r(data_glob)
            return cls(dist_gspc, data_loc.reshape((*shape, -1)))


class DistField(DistBuffer, Field, ABC):

    gspc: DistGSpace

    def __init__(self, dist_gspc: DistGSpace, data: NDArray):
        if not isinstance(dist_gspc, DistGSpace):
            raise TypeError(f"'dist_gspc' must be a '{DistGSpace}' instance. "
                            f"got type {type(dist_gspc)}")
        DistBuffer.__init__(self, dist_gspc, data)


class DistFieldG(DistField, FieldG):

    BufferType = FieldG

    def __init__(self, dist_gspc: DistGSpace, data: NDArray):
        DistField.__init__(self, dist_gspc, data)


class DistFieldR(DistField, FieldR):

    BufferType = FieldR

    def __init__(self, dist_gspc: DistGSpace, data: NDArray):
        DistField.__init__(self, dist_gspc, data)


class DistWavefun(DistBuffer, Wavefun, ABC):

    gspc: DistGkSpace
    gkspc: DistGkSpace

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        if not isinstance(dist_gkspc, DistGkSpace):
            raise TypeError(f"'gkspc' must be a '{DistGkSpace}' instance. "
                            f"got type {type(dist_gkspc)}")
        DistBuffer.__init__(self, dist_gkspc, data)
        self.gkspc = self.gspc
        self._zgemm = get_zgemm(type(data))


class DistWavefunG(DistWavefun, WavefunG):

    BufferType = WavefunG

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        DistWavefun.__init__(self, dist_gkspc, data)

    def vdot(self, ket: DistWavefunG) -> NDArray:
        out = WavefunG.vdot(self, ket)
        comm = self.gkspc.pwgrp_comm
        comm.Allreduce(comm.IN_PLACE, out, comm.SUM)
        return out


class DistWavefunR(DistWavefun, WavefunR):

    BufferType = WavefunR

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        DistWavefun.__init__(self, dist_gkspc, data)


class DistWavefunSpinG(DistWavefun, WavefunSpinG):

    BufferType = WavefunSpinG

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        DistWavefun.__init__(self, dist_gkspc, data)


class DistWavefunSpinR(DistWavefun, WavefunSpinR):

    BufferType = WavefunSpinR

    def __init__(self, dist_gkspc: DistGkSpace, data: NDArray):
        DistWavefun.__init__(self, dist_gkspc, data)
