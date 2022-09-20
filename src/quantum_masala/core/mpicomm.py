from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py.MPI import Intracomm
from typing import Any, Union

import numpy as np

from quantum_masala.config import USE_MPI, CUPY_INSTALLED

if USE_MPI:
    from mpi4py.MPI import COMM_WORLD
    from mpi4py.MPI import IN_PLACE, SUM, MIN, MAX
else:
    COMM_WORLD = None

if CUPY_INSTALLED:
    from cupy import get_array_module, ndarray
    Buffer = Union[np.ndarray, ndarray]
else:
    def get_array_module():
        return np
    Buffer = np.ndarray


class WorldComm:
    """World Communicator Module

    Mostly used for parallel routines running across all procs like:
        - Syncing Charge Densities, Local Potentials
        - Computing Occupations
        - Computing Charge-density from wavefunctions

    Methods implemented are aliases to methods of `Comm` in mpi4py.
    Note that mpi4py is an optional dependency. When running serially, nothing
    from mpi4py is imported or used in any routines.

    Attributes
    ----------
    comm : Optional[IntraComm]
        If running in parallel, `MPI.COMM_WORLD` object, else `None`.
    size : int
        Total number of procs in `WORLD`.
    rank : int
        Rank of procs in `WORLD` group.

    """

    def __init__(self):
        self.comm = None
        self.size, self.rank = 1, 0
        if USE_MPI:
            self.comm = COMM_WORLD
            self.size, self.rank = self.comm.Get_size(), self.comm.Get_rank()

    def barrier(self):
        """Alias of `Comm.barrier()`"""
        if self.size == 1:
            return
        self.comm.barrier()

    def Bcast(self, arr: Buffer):
        """Alias of `Comm.Bcast(arr)`"""
        if self.size != 1:
            self.comm.Bcast(arr)
        return arr

    def bcast(self, obj: Any):
        """Alias of `Comm.bcast(obj)`"""
        if self.size == 1:
            return obj
        return self.comm.bcast(obj)

    def allreduce_sum(self, val: Any):
        """Alias of `Comm.allreduce(val, op=SUM)`"""
        if self.size == 1:
            return val
        return self.comm.allreduce(val, op=SUM)

    def allreduce_min(self, val: Any):
        """Alias of `Comm.allreduce(val, op=MIN)`"""
        if self.size == 1:
            return val
        return self.comm.allreduce(val, op=MIN)

    def allreduce_max(self, val: Any):
        """Alias of `Comm.allreduce(val, op=MAX)`"""
        if self.size == 1:
            return val
        return self.comm.allreduce(val, op=MAX)

    def Allreduce_sum(self, arr: Buffer):
        """Alias of `Comm.Allreduce(IN_PLACE, arr)`"""
        if self.size == 1:
            return arr
        self.comm.Allreduce(IN_PLACE, arr)
        return arr


class KgrpInterComm:
    """Intercomm Module connecting all k-groups

    The group of all procs is evenly split into `numkgrp` k-groups. This module facilitates
    communication between them by making a new group containing the root proc of each k-group.
    For now, this is rarely used and is included here for completeness, so no methods are implemented
    here.

    Attributes
    ----------
    comm : Optional[Comm]
        `MPI.Comm` object generated from the group of root procs of each k-group.
    numkgrp : int
        Total number of k-groups.
    idxkgrp : int
        Index of the proc's k-group.
    size : int
        Size of the communicator group; Equals number of k-groups.
    rank : Optional[int]
        Rank of the proc in communicator group; Equals the index of the k-group the proc
        is in provided it is also the root of the group, else `None`.
    """

    def __init__(self, COMM_WORLD: Intracomm, numkgrp: int):
        self.comm = None
        self.numkgrp, self.idxkgrp = 1, 0
        self.size, self.rank = 1, 0
        if COMM_WORLD is not None:
            world_size, world_rank = COMM_WORLD.Get_size(), COMM_WORLD.Get_rank()
            self.numkgrp = numkgrp
            kgrp_size = world_size // self.numkgrp
            self.idxkgrp = world_rank // kgrp_size
            kgrp_rank = world_rank % self.size

            if self.numkgrp != 1:
                kroot_grp = COMM_WORLD.Get_group().Incl(
                    [
                        kgrp_size * ikgrp
                        for ikgrp in range(self.numkgrp)
                    ]
                )
                self.comm = COMM_WORLD.Create_group(kroot_grp)

            if kgrp_rank != 0:
                self.rank = None
            else:
                self.rank = self.idxkgrp
            self.size = self.numkgrp


class KgrpIntraComm:
    """Communicator Module for K-Group

    This Module implements required routines to perform band-parallelized diagonalization
    of the Hamiltonian and other methods that are band parallelized like:
        - Time-Propagation of KS Wavefunctions
        - Computing Wavefunction Amplitudes

    Attributes
    ----------
    size : int
        Size of the k-group containing the proc.
    rank : int
        Rank of proc within its k-group.
    comm : Optional[Comm]
        MPI 'Comm' object of the proc's k-group.
        `None` if the size of the k-group is 1.
    numkgrp : int
        Total number of k-groups
    idxkgrp : int
        Index of the proc's k-group
    """

    def __init__(self, COMM_WORLD: Intracomm, numkgrp: int):
        self.comm = None
        self.numkgrp, self.idxkgrp = 1, 0
        self.size, self.rank = 1, 0
        if COMM_WORLD is not None:
            world_size, world_rank = COMM_WORLD.Get_size(), COMM_WORLD.Get_rank()
            self.numkgrp = numkgrp
            self.size = world_size // self.numkgrp
            self.idxkgrp = world_rank // self.size
            self.rank = world_rank % self.size

            if self.size != 1:
                self.comm = COMM_WORLD.Split(self.idxkgrp, world_rank)

        self.is_root = self.rank == 0

    def split_numbnd(self, numbnd_all: int):
        """Divides the input number across all processes as evenly as possible"""
        if self.comm is None:
            return numbnd_all

        numbnd = numbnd_all // self.size
        numbnd += self.rank < (numbnd_all % self.size)
        return numbnd

    def barrier(self):
        """Alias of `Comm.barrier()`"""
        if self.size == 1:
            return
        self.comm.barrier()

    def bcast(self, obj: Any):
        """Alias of `Comm.bcast(obj)`"""
        if self.size == 1:
            return obj
        return self.comm.bcast(obj)

    def Bcast(self, arr: Buffer):
        """Alias of `Comm.Bcast(arr)`"""
        if self.size != 1:
            self.comm.Bcast(arr)
        return arr

    def allreduce_sum(self, val: Any):
        """Alias of `Comm.allreduce(val, op=SUM)`"""
        if self.size == 1:
            return val
        return self.comm.allreduce(val, op=SUM)

    def allgather(self, obj: Any):
        """Alias of `Comm.allgather(obj)`"""
        if self.size == 1:
            return [obj]
        return self.comm.allgather(obj)

    def psi_scatter_slice(self, start: int, stop: int):
        """Distributes the number of wfn's across processes in k-group as slices"""
        if self.size == 1:
            return slice(start, stop)
        numpsi = stop - start
        start += (numpsi // self.size) * self.rank + min(self.rank, numpsi % self.size)
        stop = start + (numpsi // self.size) + (self.rank < numpsi % self.size)
        return slice(start, stop)

    def psi_Allgather_inplace(self, l_psi_all: np.ndarray):
        """Performs an **in-place** 'Allgather' on input array which contains respective wfn's at right location,
        given by `psi_scatter_slice(0, l_psi_all.shape[0])`"""
        if self.size == 1:
            return l_psi_all

        numpsi_all = l_psi_all[0]

        bufspec = np.ones(self.size, dtype='i8') * (numpsi_all // self.size)
        bufspec += np.arange(self.size, dtype='i8') < (numpsi_all % self.size)
        bufspec *= np.prod(l_psi_all[1:], dtype='i8')

        self.comm.Allgatherv(IN_PLACE, [l_psi_all, bufspec])
        return l_psi_all

    def psi_Allgather(self, l_psi_bgrp: np.ndarray):
        """Gathers all wfc's across all processes in k-group"""
        if self.size == 1:
            return l_psi_bgrp

        numpsi_all = self.allreduce_sum(l_psi_bgrp.shape[0])
        xp = get_array_module(l_psi_bgrp)
        l_psi_all = xp.empty((numpsi_all, *l_psi_bgrp.shape[1:]), dtype=xp.complex128)
        sl = self.psi_scatter_slice(0, numpsi_all)
        l_psi_all[sl] = l_psi_bgrp
        self.psi_Allgather_inplace(l_psi_all)

        return l_psi_all

    def psi_Allreduce_sum(self, l_psi: np.ndarray):
        """Alias of `Comm.Allreduce_sum(IN_PLACE, l_psi)"""
        if self.size == 1:
            return l_psi
        self.comm.Allreduce(IN_PLACE, l_psi)
        return l_psi


class PWComm:
    """Communicator Module for PW

    Composition of all other Communicator Modules; Initialized at start of program and
    is passed to most objects during its initialization.

    Attributes
    ----------
    world_comm : WorldComm
        Instance of `WorldComm` containing `COMM_WORLD` communicator if program is run with `mpirun`.
    numkgrp : int
        Total number of k-groups
    kgrp_size : int
        Number of processes within a k-group
    numbgrp : int
        Number of band-groups per k-group.
        As size of a band-group is set to 1, equals the total number of processes per k-group

    kgrp_intracomm : KgrpIntraComm
        Communicator module for communication between members of k-group
    kgrp_intercomm : KgrpInterComm
        Communicator module for communication between k-groups via a group of root processes of each k-group

    is_world_root : bool
        `True` if root process of `WORLD` group i.e. `world_comm.rank == 0`
    is_kgrp_root : bool
        `True` if root process of k-group i.e `kgrp_intracomm.rank == 0`
    """

    def __init__(self, numkgrp: int = 1):
        self.world_comm = WorldComm()
        self.numkgrp = numkgrp
        if self.world_comm.size % self.numkgrp != 0:
            raise ValueError(f"number of k-groups not a factor of total number of MPI Processes\n"
                             f"Got 'world_size'={self.world_comm.size}, 'numkgrp'={self.numkgrp}")
        self.kgrp_size = self.world_comm.size // self.numkgrp

        self.kgrp_intracomm = KgrpIntraComm(self.world_comm.comm, self.numkgrp)
        self.kgrp_intercomm = KgrpInterComm(self.world_comm.comm, self.numkgrp)

        self.idxkgrp = self.kgrp_intercomm.idxkgrp
        self.kgrp_rank = self.kgrp_intracomm.rank

        self.is_world_root = self.world_comm.rank == 0
        self.is_kgrp_root = self.kgrp_intracomm.rank == 0

    def split_numk(self, numk_all: int):
        """Divides the input number across all processes as evenly as possible"""
        if self.numkgrp == 1:
            return numk_all

        numk = numk_all // self.numkgrp
        numk += self.idxkgrp < (numk_all % self.numkgrp)
        return numk

