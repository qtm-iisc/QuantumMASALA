"""Module required for implementing Parallelization in QuantumMASALA

This module implements parallelization in QuantumMASALA via MPI (``mpi4py``)
by providing classes wrapping the required MPI operations. When running in
serial, ``mpi4py`` is not imported at all, making it an optional requirement.
The module provides different parallelization 'levels', increasing the
scalability of the code.

Notes
-----
Currently, QuantumMASALA has implemented the following parallelization levels:

1. Across k-points: The group of all processes ('world') is divided into equal
groups called k-group ('kgrp'), each tasked to handle a subset of all k-points.
2. Across bands: The 'kgrp' group distribute bands among themselves, speeding
up finding eigenpairs.

Currently, each process in a 'kgrp' is assigned a subset of bands. The third
level of parallelization, involving distributing each band across a subset of
'kgrp' is yet to be implemented.
"""

__all__ = ["CommMod", "KgrpIntracomm", "PWComm"]

from importlib.util import find_spec
from typing import Optional
import numpy as np


_MPI4PY_INSTALLED = find_spec("mpi4py") is not None
"""Flag indicating if ``mpi4py`` is installed (`bool`)
"""

_CUPY_INSTALLED = find_spec("cupy") is not None
"""Flag indicating if ``cupy`` is installed (`bool`)
"""

if _CUPY_INSTALLED:
    from cupy import get_array_module
else:
    def get_array_module(_):
        return np

COMM_WORLD = None
"""Communicator for MPI World Group, `None` if running in serial
"""

# Importing required ``mpi4py`` objects
if _MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
    from mpi4py.MPI import IN_PLACE, SUM, MIN, MAX


class CommMod:
    """Communicator class wrapping ``mpi4py.MPI.Comm``

    This class wraps all MPI routines required by QuantumMASALA. Methods
    impelmented will return appropriate output when running in serial and/or
    without ``mpi4py`` installed.

    Parameters
    ----------
    comm : `Optional[mpi4py.MPI.Comm]`
        MPI Communicator. `None` if 'serial'.
    """

    def __init__(self, comm):
        self.comm = comm
        """MPI Communicator (`Optional[mpi4py.MPI.Intracomm]`).
        """

        self.size = 1
        """Size of the group (`int`).
        """

        self.rank = 0
        """Rank of the process in the group (`int`).
        """

        if comm is not None:
            self.size, self.rank = self.comm.Get_size(), self.comm.Get_rank()

    def distribute_range(self, range_, round_robin=False):
        """Distributes input range across all processes in group
        as evenly as possible

        Parameters
        ----------
        range_ : `range`
            Range to be distributed
        round_robin : `bool`, optional
            If set `True`, will return range where its elements are distributed
            in a round-robin fashion

        Returns
        -------
        The range of all indices in `range_` assigned to the process in group

        Notes
        -----
        This function does not involve any communication. Make sure all
        processes in the group call with same arguments.
        """
        if round_robin:
            return range(range_.start, range_.stop, range_.step * self.size)
        else:
            len_ = len(range_)
            start = range_.start + ((len_ // self.size) * self.rank
                                    + min(self.rank, len_ % self.size)
                                    ) * range_.step
            stop = start + (len_ // self.size
                            + self.rank < len_ % self.size) * range_.step
            return range(start, stop, range_.step)

    def barrier(self):
        """Alias of ``mpi4py.MPI.Comm.barrier()``
        """
        if self.size == 1:
            return
        self.comm.barrier()

    def bcast(self, obj):
        """Alias of ``mpi4py.MPI.Comm.bcast(obj)``
        """
        if self.size == 1:
            return obj
        return self.comm.bcast(obj)

    def Bcast(self, arr):
        """Alias of ``mpi4py.MPI.Comm.Bcast(obj)``
        """
        if self.size != 1:
            self.comm.Bcast(arr)
        return arr

    def allreduce_sum(self, val):
        """Alias of ``mpi4py.MPI.Comm.allreduce(val, op=mpi4py.MPI.SUM)``
        """
        if self.size == 1:
            return val
        return self.comm.allreduce(val, op=SUM)

    def allreduce_min(self, val):
        """Alias of ``mpi4py.MPI.Comm.allreduce(val, op=mpi4py.MPI.MIN)``
        """
        if self.size == 1:
            return val
        return self.comm.allreduce(val, op=MIN)

    def allreduce_max(self, val):
        """Alias of ``mpi4py.MPI.Comm.allreduce(val, op=mpi4py.MPI.MAX)``
        """
        if self.size == 1:
            return val
        return self.comm.allreduce(val, op=MAX)

    def Allreduce_sum_inplace(self, arr):
        """Alias of ``mpi4py.MPI.Comm.Allreduce(IN_PLACE, arr, op=mpi4py.MPI.SUM)``
        """
        if self.size != 1:
            self.comm.Allreduce(IN_PLACE, arr, op=SUM)
        return arr

    def allgather(self, obj):
        """Alias of ``mpi4py.MPI.Comm.allgather(obj)``
        """
        if self.size == 1:
            return [obj, ]
        return self.comm.allgather(obj)


class KgrpIntracomm(CommMod):
    """IntraCommunicator for k-group; required for band parallelization

    Extends ``CommMod`` with methods required for band-parallelization.

    Parameters
    ----------
    comm : `Optional[mpi4py.MPI.Comm]`
        MPI Communicator. `None` if 'serial'.

    Notes
    -----
    This class is expected to change when the next parallelization layer is
    implemented
    """

    def __init__(self, comm):
        super().__init__(comm)

    def psi_scatter_slice(self, start, stop):
        """Distributes the # of wavefunction across processes in k-group as slices

        Parameters
        ----------
        start : `int`
            First index of wavefunction block to be distributed
        stop : `int`
            Last index of wavefunction block to be distributed

        Returns
        -------
        A `slice` local to each process in k-group representing the indices of
        wavefunctions assigned to it
        """
        if self.size == 1:
            return slice(start, stop)
        numpsi = stop - start
        start += (numpsi // self.size) * self.rank + min(self.rank, numpsi % self.size)
        stop = start + (numpsi // self.size) + (self.rank < numpsi % self.size)
        return slice(start, stop)

    def psi_Allgather_inplace(self, l_psi_all: np.ndarray):
        """Performs an **in-place** Allgather on input array of wfn's, returning
        the same array but with all wfn's across the 'kgrp'

        The input array must  contain the process-local wfn's at indices specified
        by ``self.psi_scatter_slice(0, l_psi_all.shape[0])``

        Parameters
        ----------
        l_psi_all : `numpy.ndarray`, `(numpsi_all, ...)`
            Input array of wavefunctions; on input contains only the process-local
            wfn's

        Returns
        -------
        The input array now containing all wavefunctions across 'kgrp'
        """
        if self.size == 1:
            return l_psi_all

        numpsi_all = l_psi_all.shape[0]

        bufspec = np.ones(self.size, dtype='i8') * (numpsi_all // self.size)
        bufspec += np.arange(self.size, dtype='i8') < (numpsi_all % self.size)
        bufspec *= np.prod(l_psi_all.shape[1:], dtype='i8')

        self.comm.Allgatherv(IN_PLACE, [l_psi_all, bufspec])
        return l_psi_all

    def psi_Allgather(self, l_psi: np.ndarray):
        """Gathers all wfc's across all processes in k-group and return the
        aggregate to all processes in 'kgrp'

        Parameters
        ----------
        l_psi : `numpy.ndarray`
            Input array of wfn's local to each process

        Returns
        -------
        l_psi_all : `numpy.ndarray`
            Array of ``l_psi`` across all processes in 'kgrp' concatenated
            along the first axis.
        """
        if self.size == 1:
            return l_psi

        numpsi_all = self.allreduce_sum(l_psi.shape[0])
        xp = get_array_module(l_psi)
        l_psi_all = xp.empty((numpsi_all, *l_psi.shape[1:]), dtype=xp.complex128)
        sl = self.psi_scatter_slice(0, numpsi_all)
        l_psi_all[sl] = l_psi
        self.psi_Allgather_inplace(l_psi_all)

        return l_psi_all


class PWComm:
    """Container for all communicators required for PW

    Composition of all other Communicator Modules; Initialized at the start of
    program and is passed to most objects during its initialization.

    Parameters
    ----------
    numkgrp : `int`
        Number of 'kgrp' to create from 'world' group

    Raises
    ------
    ValueError
        Raised if ``numkgrp`` does not divide 'world' group evenly
    """

    def __init__(self, numkgrp: Optional[int] = None):
        self.world_comm: CommMod = CommMod(COMM_WORLD)
        """Communicator Module for 'world' group
        """

        self.world_size: int = self.world_comm.size
        """Size of 'world' group
        """

        self.world_rank: int = self.world_comm.rank
        """Rank of process in 'world' group
        """
        if numkgrp is None:
            numkgrp = self.world_size
        if self.world_comm.size % numkgrp != 0:
            raise ValueError("'numkgrp' does not divide 'world' group evenly\n"
                             f"Got 'world_size'={self.world_comm.size}, "
                             f"'numkgrp'={numkgrp}"
                             )

        self.numkgrp: int = numkgrp
        """Number of 'kgrp'
        """

        self.kgrp_size: int = self.world_size // self.numkgrp
        """Size of each 'kgrp'
        """

        self.idxkgrp: int = self.world_rank // self.kgrp_size
        """Index of the process' 'kgrp'
        """

        if COMM_WORLD is not None:
            kgrp_intracomm = COMM_WORLD.Split(self.idxkgrp, self.world_rank)
        else:
            kgrp_intracomm = None
        self.kgrp_intracomm: KgrpIntracomm = KgrpIntracomm(kgrp_intracomm)
        """Communicator module for 'kgrp'
        """

        self.kgrp_rank: int = self.kgrp_intracomm.rank
        """Rank of process in 'kgrp' group
        """

        kgrp_intercomm = None
        if self.numkgrp != 1:
            kgrp_intercomm = COMM_WORLD.Create_group(
                COMM_WORLD.Get_group().Incl(
                    [
                        self.kgrp_size * ikgrp
                        for ikgrp in range(self.numkgrp)
                    ]
                )
            )
        if self.kgrp_rank == 0:
            self.kgrp_intercomm: CommMod = CommMod(kgrp_intercomm)
        else:
            self.kgrp_intercomm: CommMod = None
