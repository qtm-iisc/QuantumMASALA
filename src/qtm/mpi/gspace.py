# from __future__ import annotations
from typing import Optional
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

        # Finding the (x, y) coordinates of all G-vectors
        nx, ny, nz = grid_shape
        ix, iy, iz = np.unravel_index(idxgrid, grid_shape, order='C')

        # Finding the unique (x, y) points; the sticks span along z-Axis
        ixy = ix * ny + iy  # Normal ordering with x and y coordinates
        ixy_sticks = np.unique(ixy)  # 'ixy_sticks' are sorted too
        numsticks = len(ixy_sticks)
        ix_sticks = ixy_sticks // ny
        iy_sticks = ixy_sticks % ny

        self.pwgrp_comm = comm
        self.pwgrp_size, self.pwgrp_rank = self.pwgrp_comm.size, self.pwgrp_comm.rank

        # Dividing the sticks across the processes
        sl = scatter_slice(numsticks, self.pwgrp_size, self.pwgrp_rank)
        ixy_sticks_loc = ixy_sticks[sl]
        numsticks_loc = len(ixy_sticks_loc)

        # Finding G-vector along the selected sticks
        # Since the G-vectors in gspc_glob are ordered lexically
        # ixy indices are already sorted in ascending order, so searchsorted is
        # adequate for the purpose
        self.ig_loc = slice(
            np.searchsorted(ixy, ixy_sticks_loc[0]),
            np.searchsorted(ixy, ixy_sticks_loc[-1], 'right')
        )

        # A work array is created to store the values along the selected sticks
        # And the FFT along Z axis is performed
        self._work_sticks = self.FFTBackend.create_buffer((nz, numsticks_loc))
        self._fftz = self.FFTBackend(self._work_sticks, (0, ))
        # Mapping selected G-vectors to the correct position in work_sticks
        # Note that although the work array is 2D, the mapping is to the
        # 1D flattened array with C-ordering
        self._g2sticks_loc = iz[self.ig_loc] * numsticks_loc + \
            np.searchsorted(ixy_sticks_loc, ixy[self.ig_loc])

        # Generating MPI Subarrays for all-to-all communication
        np_dtype = self._work_sticks.dtype
        self._sticks_subarray = gen_subarray_dtypes(
            self._work_sticks.shape, 0, np_dtype, self.pwgrp_size)

        # After transforming the sticks, the array needs to be transposed
        # So that the z-axis is split across processes.
        self.nz_loc = scatter_len(nz, self.pwgrp_size, self.pwgrp_rank)
        self.iz_loc = scatter_slice(nz, self.pwgrp_size, self.pwgrp_rank)
        self.ir_loc = np.arange(nx * ny * nz, dtype='i8').reshape((nx, ny, nz))[
            self.iz_loc.start <= iz < self.iz_loc.stop
        ]

        # The all-to-all communication redistributes the global (nz, numsticks)
        # array such that the local arrays that are slices across first dimension
        # are transformed to arrays that are slices across second dimension
        # Creating a transfer array for communication
        self._work_transfer = self.FFTBackend.create_buffer((self.nz_loc, numsticks))
        self._transfer_subarray = gen_subarray_dtypes(
            self._work_transfer.shape, 1, np_dtype, self.pwgrp_size
        )

        # After the data is received, the sticks, now split acriss its length
        # are placed at their respective sited and the FFT is performed
        # across the XY plane (which are the last two dimensions; the fastest)
        self._work_full = self.FFTBackend.create_buffer((self.nz_loc, nx, ny))
        self._fftxy = self.FFTBackend(self._work_full, (1, 2))
        # The data in transfer array is placed to the 3D slab work array
        # based on the coordinates of the sticks
        self._transfer2full = ix_sticks * ny + iy_sticks

        # Now the GSpaceBase is called with the subset of G-vectors assigned
        # to the process. We have replaced the FFT3D Class with a
        # dummy one, so that the parent __init__ will not have a valid FFT3D
        # instance. We need to overload the corresponding methods
        GSpaceBase.__init__(self, self.gspc_glob.recilat,
                            self.gspc_glob.grid_shape,
                            self.gspc_glob.g_cryst[:, self.ig_loc])

        # Some attributes already set by super().__init__ are overwritten as it is
        # incorrect/invalid
        self.grid_shape = (nx, ny, self.nz_loc)
        self.size_r = int(np.prod(self.grid_shape))
        # 'GSpaceBase' attributes that are disabled as they are not defined
        # when distributed
        self.idxgrid, self.idxsort = None, None

        # The G-vectors in the global version are simply split to appropriate lengths
        # and assigned to each process. So scatter/gather processes simply need
        # buffer send/recv counts
        self._scatter_g_bufspec = []
        for rank in range(self.pwgrp_size):
            sl = scatter_slice(numsticks, self.pwgrp_size, self.pwgrp_rank)
            ixy_sticks_loc = ixy_sticks[sl]
            ig_start = np.searchsorted(ixy, ixy_sticks_loc[0])
            ig_stop = np.searchsorted(ixy, ixy_sticks_loc[-1], 'right')
            self._scatter_g_bufspec.append(ig_stop - ig_start)

        # For the real-space, since the 3D FFT array is split across the last dimension
        # the data to be transferred is not contiguous, hence we need a MPI_Vector
        # Datatype for the communication
        self._scatter_r_sendbuf = (
            scatter_len(nz, self.pwgrp_size),
            gen_vector_dtype(self.gspc_glob.grid_shape, 2, np_dtype)
        )
        self._scatter_r_recvbuf = (
            gen_vector_dtype(self.grid_shape, 2, np_dtype),
        )

    def _r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Similar to FFT3DSticks but with communication between the two FFT
        self._work_full[:] = arr_inp.transpose((2, 0, 1))
        self._fftxy.fft()

        work_full = self._work_full.reshape((self.nz_loc, -1))
        work_full.take(self._transfer2full, axis=1, out=self._work_transfer)

        self.pwgrp_comm.comm.Alltoallw(
            (self._work_transfer, self._transfer_subarray),
            (self._work_sticks, self._sticks_subarray)
        )

        self._fftz.fft()
        self._work_sticks.take(self._g2sticks_loc, out=arr_out)

    def _g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Similar to FFT3DSticks but with communication between the two FFT
        self._work_sticks.fill(0)
        self._work_sticks.reshape(-1)[self._g2sticks_loc] = arr_inp
        self._fftz.ifft(self._normalise_idft)

        self.pwgrp_comm.comm.Alltoallw(
            (self._work_sticks, self._sticks_subarray),
            (self._work_transfer, self._transfer_subarray)
        )

        self._work_full.fill(0)
        self._work_full.reshape((self.nz_loc, -1))[
            (slice(None), self._transfer2full)
        ] = self._work_transfer
        self._fftxy.ifft(self._normalise_idft)

        arr_out[:] = self._work_full.transpose((1, 2, 0))

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
                (sendbuf[iarr], *self._scatter_r_sendbuf)
                if self.pwgrp_rank == 0 else None,
                (recvbuf[iarr], *self._scatter_r_recvbuf)
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
                    (recvbuf[iarr], *self._scatter_g_sendbuf)
                )
            else:
                self.pwgrp_comm.comm.Gatherv(
                    (sendbuf[iarr], *self._scatter_r_recvbuf),
                    (recvbuf[iarr], *self._scatter_r_sendbuf)
                    if is_root or allgather else None
                )
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
        recvbuf = out.reshape((-1, self.size_r))

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
