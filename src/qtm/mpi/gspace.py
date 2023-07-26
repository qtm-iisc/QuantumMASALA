# from __future__ import annotations
from typing import Optional
from qtm.config import NDArray
__all__ = ['DistGSpaceBase', 'DistGSpace', 'DistGkSpace']

import numpy as np

from qtm.gspace import GSpaceBase, GSpace, GkSpace
from qtm.mpi import QTMComm

from qtm.mpi.utils import scatter_range, scatter_len


# Creating a Dummy FFT3D class so that no FFT planning is done in GSpaceBase.__init__
# As DistGSpace implements distributed FFT operations
class DummyFFT3D:

    def __init__(self, *args, **kwargs):
        pass


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

        # Finding the (y, z) coordinates of all G-vectors
        nx, ny, nz = grid_shape
        ix, iy, iz = np.unravel_index(idxgrid, grid_shape, order='C')

        # Finding the unique (y, z) points; the sticks span along x-Axis
        iyz = iy * nz + iz  # Normal ordering with y and z coordinates
        iyz_sticks = np.unique(iyz)  # 'iyz_sticks' are sorted too
        numsticks = len(iyz_sticks)
        iy_sticks = iyz_sticks // nz
        iz_sticks = iyz_sticks % nz

        self.pwgrp_comm = comm
        self.pwgrp_size, self.pwgrp_rank = self.pwgrp_comm.size, self.pwgrp_comm.rank

        # Dividing the sticks across the processes
        range_loc = scatter_range(range(numsticks), self.pwgrp_size, self.pwgrp_rank)
        sticks_start, sticks_stop = range_loc.start, range_loc.stop
        numsticks_loc = sticks_stop - sticks_start
        iyz_sticks_loc = iyz_sticks[sticks_start:sticks_stop]

        # Finding G-vector along the selected sticks
        self.ig_loc = np.nonzero(
            (iyz >= iyz_sticks_loc[0]) * (iyz <= iyz_sticks_loc[-1])
        )[0]

        # A work array is created to store the values along the selected sticks
        # And the FFT along X axis is performed
        self._work_sticks = self.FFTBackend.create_buffer((nx, numsticks_loc))
        self._fftx = self.FFTBackend(self._work_sticks, (0, ))
        # Mapping selected G-vectors to the correct position in work_sticks
        # Note that although the work array is 2D, the mapping is to the
        # 1D flattened array with C-ordering
        self._g2sticks_loc = ix[self.ig_loc] * numsticks_loc \
            + np.searchsorted(iyz_sticks_loc, iyz[self.ig_loc])

        # After transforming the sticks, the array needs to be transposed
        # So that the x-axis is split across processes.
        nx_loc = scatter_len(nx, self.pwgrp_size, self.pwgrp_rank)

        # The MPI communication step involves calling MPI_Alltoallv instead
        # of creating a MPI Subarray type and using MPI_Alltoallw. Don't ask
        # why I implemented the former. Refer to the '_transpose' method
        # for the implementation of global array transposition

        # Creating a transfer array for communication
        self._work_transfer = self.FFTBackend.create_buffer((numsticks, nx_loc))
        # After the data is received, the sticks, now split along its lenght
        # are placed at their respective sited and the FFT is performed
        # across the YZ plane (which are the last two dimensions; the fastest)
        self._work_full = self.FFTBackend.create_buffer((nx_loc, ny, nz))
        self._fftyz = self.FFTBackend(self._work_full, (1, 2))
        # The data in transfer array is placed to the 3D slab work array
        # based on the coordinates of the sticks
        self._transfer2full = iy_sticks * nz + iz_sticks

        # Now the GSpaceBase is called with the subset of G-vectors assigned
        # to the process. Note that we have replaced the FFT3D Class with a
        # dummy one, so that the parent __init__ will not have a valid FFT3D
        # instance. We need to overload the corresponding methods
        GSpaceBase.__init__(self, self.gspc_glob.recilat,
                            self.gspc_glob.grid_shape,
                            self.gspc_glob.g_cryst[:, self.ig_loc])

        # Some extra attributes for convenience
        self.nx, self.nx_loc = nx, nx_loc
        self.nyz = ny * nz
        # Some attributes already set by super().__init__ are overwritten as it is
        # incorrect/invalid
        self.grid_shape = (self.nx_loc, ny, nz)
        self.size_r = int(np.prod(self.grid_shape))
        # 'GSpaceBase' attributes that are disabled as they are not defined
        # when distributed
        self.idxgrid, self.idxsort = None, None

        # When reconstructing the serial buffer from parallel instances, we need a
        # map to rearrange the gathered (scattered) G-vectors to match the
        # serial (parallel) ordering
        iyz_key = iyz_sticks[np.cumsum(scatter_len(numsticks, self.pwgrp_size))[:-1]]
        # IMPORTANT: STABLE SORT ONLY
        self._glob2gather = np.argsort(np.searchsorted(iyz_key, iyz, 'right'),
                                       kind='stable')
        # This one doesn't require it as its input does not have identical entries
        self._gather2glob = np.argsort(self._glob2gather)

    def _transpose(self, inp: NDArray, out: NDArray):
        # Starting with an array whose rows are distributed across processes
        # MPI_Alltoallv will allow transposition of a (A_glob, B_glob) global
        # array to a (B_glob, A_glob) array again with its rows distributed
        # Each process distributes its (A_loc, B_glob) array and
        # receives data to build a (B_loc, A_glob) array
        # But the order of data received is not correct. Data chunk received
        # from each process needs to be transformed by transposition
        # before concatenating in this method.
        glob_shape = (inp.shape[0], out.shape[0])
        size, rank = self.pwgrp_size, self.pwgrp_rank
        sendcount = scatter_len(glob_shape[0], size) \
            * scatter_len(glob_shape[1], size, rank)
        recvcount = scatter_len(glob_shape[1], size) \
            * scatter_len(glob_shape[0], size, rank)

        self.pwgrp_comm.Alltoallv((inp, sendcount), (out.ravel(), recvcount))
        chunks = tuple(
            chunk.reshape((scatter_len(glob_shape[0], size, rank), -1)).T
            for chunk in np.split(out.ravel(), np.cumsum(recvcount[:-1]))
        )
        np.concatenate(chunks, axis=0, out=out)
        # TODO: Need to benchmark this with the Alltoallw + Subarray method
        # Also, I would be more inclined to the alternative if I can
        # transpose the transfer array in the same shot

    def _r2g(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Similar to FFT3DSticks but with communication between the two FFT
        self._work_full[:] = arr_inp
        self._fftyz.fft()

        work_full = self._work_full.reshape((self.nx_loc, -1)).T
        work_full.take(self._transfer2full, axis=0, out=self._work_transfer)
        self._transpose(self._work_transfer, self._work_sticks)

        self._fftx.fft()
        self._work_sticks.take(self._g2sticks_loc, out=arr_out)

    def _g2r(self, arr_inp: NDArray, arr_out: NDArray) -> None:
        # Similar to FFT3DSticks but with communication between the two FFT
        self._work_sticks.fill(0)
        self._work_sticks.reshape(-1)[self._g2sticks_loc] = arr_inp
        self._fftx.ifft(self._normalise_idft)

        self._transpose(self._work_sticks, self._work_transfer)

        self._work_full.fill(0)
        self._work_full.reshape((self.nx_loc, -1))[
            (slice(None), self._transfer2full)
        ] = self._work_transfer.T
        self._fftyz.ifft(self._normalise_idft)

        arr_out[:] = self._work_full

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

    def scatter_data_r(self, arr_root=None):
        shape, sendbuf = None, None
        if self.pwgrp_rank == 0:
            shape = arr_root.shape[:-1]
            shape = self.pwgrp_comm.bcast(shape)
            sendbuf = arr_root.reshape((-1, self.gspc_glob.size_r))
        else:
            shape = self.pwgrp_comm.bcast(shape)
        sendcounts = scatter_len(self.nx, self.pwgrp_size) * self.nyz

        out = self.create_buffer_r(shape)
        recvbuf = out.reshape((-1, self.size_r))

        for iarr in range(np.prod(shape)):
            self.pwgrp_comm.comm.Scatterv(
                (sendbuf[iarr], sendcounts) if self.pwgrp_rank == 0  else None,
                recvbuf[iarr]
            )

        return out


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
