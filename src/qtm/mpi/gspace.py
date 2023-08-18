# from __future__ import annotations
from qtm.typing import Optional
from qtm.config import NDArray
__all__ = ['DistGSpaceBase', 'DistGSpace', 'DistGkSpace']

import numpy as np

from qtm.gspace import GSpaceBase, GSpace, GkSpace
from qtm.gspace.fft.utils import DummyFFT3D
from qtm.mpi import QTMComm

from qtm.mpi.utils import (
    scatter_slice, scatter_len,
    gen_subarray_dtypes, gen_vector_dtype
)


class DistGSpaceBase(GSpaceBase):

    FFT3D = DummyFFT3D

    def __init__(self, comm: QTMComm, gspc: GSpaceBase):
        if not isinstance(comm, QTMComm):
            raise TypeError(f"'comm' must be a '{QTMComm}' instance. "
                            f"got type {type(comm)}.")
        if not isinstance(gspc, GSpaceBase):
            raise TypeError(f"'gspc' must be a '{GSpaceBase}' instance. "
                            f"got type {type(gspc)}.")

        # Referencing attributes from the serial instance 'gspc'
        self.gspc_glob = gspc
        self.FFTBackend = gspc._fft.FFTBackend
        self._normalise_idft = gspc._fft.normalise_idft
        self._normalilse_fac = gspc._fft.normalise_fac
        grid_shape = gspc.grid_shape
        idxgrid = gspc.idxgrid

        # Finding the position of G-vectors in the 3D grid
        # NOTE: G-vectors in 'GSpaceBase' instances are sorted lexically
        # with (y, z, x) coordinates. The sticks are along the X - Axis while
        # the YZ planes are distributed across processes
        nx, ny, nz = grid_shape
        ix, iy, iz = np.unravel_index(idxgrid, grid_shape, order='C')

        # Finding the unique (y, z) points; the sticks span along x-Axis
        iyz = iy * nz + iz  # 'iyz' is already sorted, according to 'GSpaceBase' init
        iyz_sticks = np.unique(iyz)
        numsticks = len(iyz_sticks)
        iy_sticks = iyz_sticks // nz
        iz_sticks = iyz_sticks % nz

        self.pwgrp_comm = comm
        self.pwgrp_size, self.pwgrp_rank = self.pwgrp_comm.size, self.pwgrp_comm.rank

        # Dividing the sticks across the processes
        sl = scatter_slice(numsticks, self.pwgrp_size, self.pwgrp_rank)
        iyz_sticks_loc = iyz_sticks[sl]
        self.numsticks_loc = len(iyz_sticks_loc)

        # Finding G-vector along the selected sticks
        # As iyz is already sorted, 'searchsorted' is applicable here
        self.ig_loc = slice(
            np.searchsorted(iyz, iyz_sticks_loc[0]),
            np.searchsorted(iyz, iyz_sticks_loc[-1], 'right')
        )
        # Definind the distribution of the 3D real-space arrays too
        self.nx_loc = scatter_len(nx, self.pwgrp_size, self.pwgrp_rank)
        self.ix_loc = scatter_slice(nx, self.pwgrp_size, self.pwgrp_rank)
        self.ir_loc = slice(self.ix_loc.start * ny * nz,
                            self.ix_loc.stop * ny * nz)

        # FFT along X axis is planned for the sticks local to process
        self._fftx = self.FFTBackend((nx, self.numsticks_loc), (0, ))
        self._fftx.inp_bwd[:] = 0
        # Mapping selected G-vectors to the correct position in work_sticks
        # Note that although the work array is 2D, the mapping is to the
        # 1D flattened array with C-ordering
        self._g2sticks_loc = ix[self.ig_loc] * self.numsticks_loc + \
            np.searchsorted(iyz_sticks_loc, iyz[self.ig_loc])

        # The sticks undergo a global transformation so that the data across
        # the X-Axis is now distributed across processes.
        # A work array is created to store this globally transposed data
        self._work_trans = self.FFTBackend.create_buffer((self.nx_loc * numsticks, ))
        # Generating 'bufspec's for 'Alltollv' communication where the data along
        # sticks are distributed
        self._sticks_bufspecv = scatter_len(nx, self.pwgrp_size) * self.numsticks_loc
        self._trans_bufspecv = scatter_len(numsticks, self.pwgrp_size) * self.nx_loc

        # Now the data in transfer array is arranged in the 3D array that is
        # a local slice of the FFT grid, cut across X-Acis
        self._trans2full = iy_sticks * nz + iz_sticks
        self._fftyz = self.FFTBackend((self.nx_loc, ny, nz), (1, 2))
        self._fftyz.inp_bwd[:] = 0

        # Now the GSpaceBase is called with the subset of G-vectors assigned
        # to the process. We have replaced the FFT3D Class with a
        # dummy one, so that the parent __init__ will not have a valid FFT3D
        # instance. We need to overload the corresponding methods
        GSpaceBase.__init__(self, self.gspc_glob.recilat,
                            self.gspc_glob.grid_shape,
                            self.gspc_glob.g_cryst[:, self.ig_loc])

        # Some attributes already set by super().__init__ are overwritten as it is
        # incorrect/invalid
        self.grid_shape = (self.nx_loc, ny, nz)
        self.size_r = int(np.prod(self.grid_shape))
        # 'GSpaceBase' attributes that are disabled as they are not defined
        # when distributed
        self.idxgrid, self.idxsort = None, None

        # The G-vectors in the global version are simply split to appropriate lengths
        # and assigned to each process. So scatter/gather processes simply need
        # buffer send/recv counts
        self._scatter_g_bufspec = []
        for rank in range(self.pwgrp_size):
            sl = scatter_slice(numsticks, self.pwgrp_size, rank)
            iyz_sticks_loc = iyz_sticks[sl]
            ig_start = np.searchsorted(iyz, iyz_sticks_loc[0])
            ig_stop = np.searchsorted(iyz, iyz_sticks_loc[-1], 'right')
            self._scatter_g_bufspec.append(ig_stop - ig_start)

        # For the real-space, the 3D FFT array is split across the first dimension
        # which yield contiguous chunks, so the size of each chunk is computed
        self._scatter_r_bufspec = scatter_len(nx, self.pwgrp_size) * ny * nz

    def _r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Similar to FFT3DSticks but with communication between the two FFT
        self._fftyz.inp_fwd[:] = arr_inp
        work_full = self._fftyz.fft().reshape((self.nx_loc, -1))
        work_trans = self._work_trans.reshape((-1, self.nx_loc))
        work_full.take(self._trans2full, axis=1, out=work_trans.T)

        work_sticks = self._fftx.inp_fwd.ravel()
        self.pwgrp_comm.comm.Alltoallv(
            (work_trans, self._trans_bufspecv),
            (work_sticks, self._sticks_bufspecv)
        )
        np.concatenate(tuple(
            arr.reshape(self.numsticks_loc, -1).T for arr in
            np.split(work_sticks, np.cumsum(self._sticks_bufspecv)[:-1])
        ), axis=0, out=work_sticks.reshape((-1, self.numsticks_loc)))

        work_sticks = self._fftx.fft()
        work_sticks.take(self._g2sticks_loc, out=arr_out)

    def _g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Similar to FFT3DSticks but with communication between the two FFT
        work_sticks = self._fftx.inp_bwd
        work_sticks.reshape(-1)[self._g2sticks_loc] = arr_inp
        work_sticks = self._fftx.ifft(self._normalise_idft)

        work_trans = self._work_trans
        self.pwgrp_comm.comm.Alltoallv(
            (work_sticks, self._sticks_bufspecv),
            (work_trans, self._trans_bufspecv)
        )
        np.concatenate(tuple(
            arr.reshape((self.nx_loc, -1)).T for arr in
            np.split(work_trans, np.cumsum(self._trans_bufspecv)[:-1])
        ), axis=0, out=work_trans.reshape((-1, self.nx_loc)))
        work_trans = work_trans.reshape((-1, self.nx_loc))

        work_full = self._fftyz.inp_bwd
        work_full.reshape((self.nx_loc, -1))[:, self._trans2full] = work_trans.T
        arr_out[:] = self._fftyz.ifft(self._normalise_idft)

    def create_buffer(self, shape: tuple[int, ...]) -> NDArray:
        """Modified to prevent accessing the now-DummyFFT instance"""
        return self.FFTBackend.create_buffer(shape)

    def check_buffer(self, arr: NDArray) -> None:
        """Modified to prevent accessing the now-DummyFFT instance"""
        self.FFTBackend.check_buffer(arr)

    def r2g(self, arr_r: NDArray, arr_g: Optional[NDArray] = None) -> NDArray:
        self.check_buffer_r(arr_r)
        if arr_g is not None:
            self.check_buffer_g(arr_g)
        else:
            arr_g = self.create_buffer_g(arr_r.shape[:-1])

        for inp, out in zip(arr_r.reshape(-1, *self.grid_shape),
                            arr_g.reshape(-1, self.size_g)):
            self._r2g(inp, out)

        return arr_g

    def g2r(self, arr_g: NDArray, arr_r: Optional[NDArray] = None) -> NDArray:
        self.check_buffer_g(arr_g)
        if arr_r is not None:
            self.check_buffer_r(arr_r)
        else:
            arr_r = self.create_buffer_r(arr_g.shape[:-1])

        for inp, out in zip(arr_g.reshape(-1, self.size_g),
                            arr_r.reshape(-1, *self.grid_shape)):
            self._g2r(inp, out)

        return arr_r

    def check_gspc_glob(self, gspc_glob: GSpaceBase):
        if gspc_glob != self.gspc_glob:
            raise ValueError("input 'gspc_glob' at root process of pwgrp "
                             "does not match local instance. ")

    def scatter_r(self, arr_root: Optional[NDArray]) -> NDArray:
        shape, sendbuf = None, None
        if self.pwgrp_rank == 0:
            self.gspc_glob.check_buffer_r(arr_root)
            shape = arr_root.shape[:-1]
            sendbuf = arr_root.reshape((-1, self.gspc_glob.size_r))
        shape = self.pwgrp_comm.bcast(shape)

        out = self.create_buffer_r(shape)
        recvbuf = out.reshape((-1, self.size_r))

        for iarr in range(np.prod(shape)):
            self.pwgrp_comm.comm.Scatterv(
                (sendbuf[iarr], self._scatter_r_bufspec)
                if self.pwgrp_rank == 0 else None,
                recvbuf[iarr]
            )
        return out

    def gather_r(self, arr_loc: NDArray, allgather: bool = False) -> Optional[NDArray]:
        self.check_buffer_r(arr_loc)
        if not isinstance(allgather, bool):
            raise TypeError("'allgather' must be a boolean. "
                            f"got '{type(allgather)}'.")

        is_root = self.pwgrp_rank == 0
        shape = self.pwgrp_comm.bcast(arr_loc.shape[:-1])
        if arr_loc.shape[:-1] != shape:
            raise ValueError("shape of 'arr_loc' inconsistent across MPI processes. "
                             f"got shape[:-1]={arr_loc.shape[:-1]} locally, but "
                             f"at root, it is {shape}.")
        sendbuf = arr_loc.reshape((-1, self.size_r))
        arr_glob, recvbuf = None, None
        if is_root or allgather:
            arr_glob = self.gspc_glob.create_buffer_r(shape)
            recvbuf = arr_glob.reshape((-1, self.gspc_glob.size_r))
        if allgather:
            recvbuf[..., self.ir_loc] = sendbuf

        for iarr in range(np.prod(shape)):
            if allgather:
                self.pwgrp_comm.comm.Allgatherv(
                    self.pwgrp_comm.IN_PLACE,
                    (recvbuf[iarr], self._scatter_r_bufspec)
                )
            else:
                self.pwgrp_comm.comm.Gatherv(sendbuf[iarr],
                                             (recvbuf[iarr], self._scatter_r_bufspec)
                                             if is_root or allgather else None)
        if is_root or allgather:
            return arr_glob

    def allgather_r(self, arr_loc: NDArray) -> NDArray:
        return self.gather_r(arr_loc, True)

    def scatter_g(self, arr_root: Optional[NDArray]):
        shape, sendbuf = None, None
        if self.pwgrp_rank == 0:
            shape = arr_root.shape[:-1]
            sendbuf = arr_root.reshape((-1, self.gspc_glob.size_g))
        shape = self.pwgrp_comm.bcast(shape)

        out = self.create_buffer_g(shape)
        recvbuf = out.reshape((-1, self.size_g))

        for iarr in range(np.prod(shape)):
            self.pwgrp_comm.comm.Scatterv(
                (sendbuf[iarr], self._scatter_g_bufspec)
                if self.pwgrp_rank == 0 else None,
                recvbuf[iarr]
            )
        return out

    def gather_g(self, arr_loc: NDArray, allgather: bool = False) -> Optional[NDArray]:
        self.check_buffer_g(arr_loc)
        if not isinstance(allgather, bool):
            raise TypeError("'allgather' must be a boolean. "
                            f"got '{type(allgather)}'.")

        is_root = self.pwgrp_rank == 0
        shape = self.pwgrp_comm.bcast(arr_loc.shape[:-1])
        if arr_loc.shape[:-1] != shape:
            raise ValueError("shape of 'arr_loc' inconsistent across MPI processes. "
                             f"got shape[:-1]={arr_loc.shape[:-1]} locally, but "
                             f"at root, it is {shape}.")
        sendbuf = arr_loc.reshape((-1, self.size_g))
        arr_glob, recvbuf = None, None
        if is_root or allgather:
            arr_glob = self.gspc_glob.create_buffer_g(shape)
            recvbuf = arr_glob.reshape((-1, self.gspc_glob.size_g))
        if allgather:
            recvbuf[..., self.ig_loc] = sendbuf

        for iarr in range(np.prod(shape)):
            if allgather:
                self.pwgrp_comm.comm.Allgatherv(
                    self.pwgrp_comm.IN_PLACE,
                    (recvbuf[iarr], self._scatter_g_bufspec)
                )
            else:
                self.pwgrp_comm.comm.Gatherv(
                    sendbuf[iarr],
                    (recvbuf[iarr], self._scatter_g_bufspec)
                    if is_root or allgather else None
                )
        if is_root or allgather:
            return arr_glob

    def allgather_g(self, arr_loc: NDArray) -> NDArray:
        return self.gather_g(arr_loc, True)


class DistGSpace(DistGSpaceBase, GSpace):

    gspc_glob: GSpace

    def __init__(self, comm: QTMComm, gspc: GSpace):
        DistGSpaceBase.__init__(self, comm, gspc)
        self.ecut = self.gspc_glob.ecut


class DistGkSpace(DistGSpaceBase, GkSpace):

    gkspc_glob: GkSpace

    def __init__(self, comm: QTMComm, gkspc: GkSpace):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(f"'gkspc' must be a '{GkSpace}' instance. "
                            f"got '{type(gkspc)}'.")
        self.gkspc_glob = gkspc

        DistGSpaceBase.__init__(self, comm, self.gkspc_glob)
        self.gspc_glob = None
        self.ecutwfn = self.gkspc_glob.ecutwfn
        self.k_cryst = self.gkspc_glob.k_cryst
        self.idxgk = None
