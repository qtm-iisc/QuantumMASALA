from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Any
__all__ = ['QTMComm', 'QTMComm',
           'BufSpec', 'BufSpecV',
           'split_comm_pwgrp']

from collections.abc import Sequence
from types import MethodType

from qtm import qtmconfig
if qtmconfig.mpi4py_installed:
    from mpi4py.MPI import COMM_WORLD, COMM_NULL, COMM_SELF
    from mpi4py.MPI import Op, IN_PLACE
    InPlace = type(IN_PLACE)
    from mpi4py.MPI import SUM, PROD, MIN, MAX, LAND, LOR, IDENT, SIMILAR

    WORLD_SIZE, WORLD_RANK = COMM_WORLD.Get_size(), COMM_WORLD.Get_rank()
    # WARNING: Following import is only done when mpi4py is installed.
    from mpi4py.MPI import Group, Intracomm

else:
    COMM_WORLD, COMM_NULL, COMM_SELF = 'COMM_WORLD', 'COMM_NULL', 'COMM_SELF'
    Intracomm, Op, InPlace = 'Intracomm', 'Op', 'InPlace'
    IN_PLACE, SUM, PROD, MIN, MAX, LAND, LOR = (
        'IN_PLACE', 'SUM', 'PROD', 'MIN', 'MAX', 'LAND', 'LOR'
    )

    WORLD_SIZE, WORLD_RANK = 1, 0
    Group = None

from qtm.config import NDArray
BufSpec = NDArray
BufSpecV = tuple[NDArray, Sequence[int]]


class QTMComm:

    mpi4py_installed = qtmconfig.mpi4py_installed
    comm_world = COMM_WORLD
    comm_null = COMM_NULL
    comm_self = COMM_SELF

    world_size: int = WORLD_SIZE
    """Size of the `mpi4py.MPI.COMM_WORLD` group"""
    world_rank: int = WORLD_RANK
    """Rank of the process in `mpi4py.MPI.COMM_WORLD`"""
    IN_PLACE = IN_PLACE
    """Alias of `mpi4py.MPI.IN_PLACE`"""
    SUM = SUM
    """Alias of `mpi4py.MPI.SUM`"""
    PROD = PROD
    """Alias of `mpi4py.MPI.PROD`"""
    MIN = MIN
    """Alias of `mpi4py.MPI.MIN`"""
    MAX = MAX
    """Alias of `mpi4py.MPI.MAX`"""
    LAND = LAND
    """Alias of `mpi4py.MPI.LAND`"""
    LOR = LOR
    """Alias of `mpi4py.MPI.LOR`"""

    def __init__(self, comm: Intracomm | None,
                 parent_comm: QTMComm | None = None,
                 sync_with_parent: bool = True):
        # Validating 'comm'
        is_null = (comm == self.comm_null)
        if not is_null:
            if self.mpi4py_installed and not isinstance(comm, Intracomm):
                raise TypeError("'comm' must be a MPI Intracommunicator. "
                                f"got {type(comm)}.")
        else:
            comm = None
        self.is_null: bool = is_null
        """True if `comm` is `mpi4py.MPI.COMM_NULL`."""

        self.comm: Intracomm | None = comm
        """MPI Intracommunicator. Will be `None` if `qtm.qtmconfig.mpi4py_installed`
        is `False`.
        """

        size, rank = 1, 0
        if self.is_null:
            size = 0
        elif self.mpi4py_installed:
            size, rank = self.comm.Get_size(), self.comm.Get_rank()
        self.size: int = size
        """Size of the group associated to `comm`.
        """
        self.rank: int = rank
        """Rank of the process in the group associated to `comm`.
        """

        if parent_comm is not None:
            if not isinstance(parent_comm, QTMComm):
                raise TypeError(
                    "If not 'None, 'parent_comm' must be an instance of "
                    f"'{QTMComm}' instance. got {type(parent_comm)}."
                )
            if parent_comm.is_null:
                raise ValueError("'parent_comm' is a null communicator i.e "
                                 "parent_comm.is_null = True. ")
            elif self.world_size == 1:
                parent_comm = None

        if not self.is_null and parent_comm is not None:
            if self.mpi4py_installed:
                comm_group = self.comm.Get_group()
                grp_intersection = Group.Intersection(
                    comm_group, parent_comm.comm.Get_group())
                if Group.Compare(comm_group, grp_intersection) not in [IDENT, SIMILAR]:
                    raise ValueError("'comm' is not a subgroup of group associated to"
                                     "'parent_comm'.")

        self.parent_comm: QTMComm | None = parent_comm
        """Parent Communicator; used to synchronize subgroups (of which this
        instance is a part of) when exiting context."""

        if not isinstance(sync_with_parent, bool):
            raise ValueError("'sync_with_parent' must be a boolean. "
                             f"got {type(sync_with_parent)}")
        if self.parent_comm is None:
            sync_with_parent = False
        self.sync_with_parent: bool = sync_with_parent
        """If True, call ``parent_comm.barrier()`` when exiting context."""

    def __getattribute__(self, item):
        """Overloaded to disable all methods if instance is a
        null communicator i.e `is_null` is True"""
        out = super().__getattribute__(item)
        if not isinstance(out, MethodType):
            return out
        elif item == 'skip_with_block':
            return out
        elif self.is_null:
            raise AttributeError("'CommMod' instance is a null communicator. "
                                 "All methods are disabled")
        return out

    def __enter__(self) -> QTMComm:
        """When used as a context manager, ``MPI_Barrier`` is called on entry
        to get all processes in the group to sync."""
        if not self.is_null:
            self.Barrier()
        return self

    class SkipWithBlock(Exception):
        pass

    def skip_with_block(self):
        raise self.SkipWithBlock

    def __exit__(self, exc_type, exc_val, exc_tb):
        """When exiting the context, all processes in group will be
        synchronized with ``MPI_Barrier`` call. This is disabled if
        `sync_with_parent` is set to `False`."""
        if isinstance(exc_val, Exception):
            if exc_type is not self.SkipWithBlock:
                raise Exception(
                    f"Process # {self.world_rank} / {self.world_size} has encountered "
                    f"an exception. Please refer above for further info."
                ) from exc_val

        # Synchronize with all processes in group
        if not self.is_null:
            self.Barrier()
        # Synchronize all processes in the parent group if required
        if self.sync_with_parent:
            self.parent_comm.Barrier()
        # If block is skipped, the exception raised for the mechanism needs to be
        # suppressed
        if exc_type is self.SkipWithBlock:
            return True

    def Barrier(self) -> None:  # noqa: N802
        if self.mpi4py_installed:
            self.comm.Barrier()

    def barrier(self) -> None:  # noqa: N802
        if self.mpi4py_installed:
            self.comm.barrier()

    def Incl(self, ranks: Sequence[int], sync_with_parent: bool = True) -> QTMComm:  # noqa: N802
        """Creates a subgroup of communicator containing only the processes
        given by `ranks`.

        All processes within the subgroup must call with
        identical arguments. Refer to `MPI_Comm_create_group` and
        `MPI_Group_incl` for further info.

        Parameters
        ----------
        ranks : Sequence[int]
            Integers represents the ranks (relative to the instance's group) of
            processes to create the subgroup with
        sync_with_parent : bool, default=True
            If True, when used in a `with` block, calls `parent_comm.Barrier()`
            so that all processes in the parent group will exit the `with`
            code block at the same time
        """
        if self.size == 1:
            if list(ranks) != [0, ]:
                raise ValueError("communicator has only one process")
            return self
        comm_out = self.comm.Create_group(
            self.comm.Get_group().Incl(ranks)
        )
        return QTMComm(comm_out, self, sync_with_parent)

    def Root(self):
        return QTMComm(self.comm_self if self.rank == 0 else self.comm_null,
                       self, sync_with_parent=True)

    def Split(self, color: int, key: int,
              sync_with_parent: bool = False) -> QTMComm:  # noqa: N802
        """Divides the group into multiple subgroups defined by `color` and
        `key` values.

        All processes in the group only must call with
        identical arguments. Refer to `MPI_Comm_split` for further info.

        Parameters
        ----------
        color : int
            Index of the subgroup to assign this process to.
        key : int
            Order/Rank of this process within the subgroup given by `color`.
        sync_with_parent : bool, default=False
            If True, when used in a `with` block, calls `parent_comm.Barrier()`
            so that all processes in the parent group will exit the `with`
            code block at the same time
        """
        comm_split = self.comm.Split(color, key)
        return QTMComm(comm_split, self, sync_with_parent)

    def bcast(self, obj, root: int = 0):
        """Alias of ``mpi4py.MPI.Comm.bcast``"""
        if self.mpi4py_installed:
            return self.comm.bcast(obj, root)
        return obj

    def scatter(self, sendobj: Sequence[Any], root: int = 0):
        if self.mpi4py_installed:
            return self.comm.scatter(sendobj, root)
        return sendobj

    def allgather(self, obj):
        """Alias of ``mpi4py.MPI.Comm.allgather``"""
        if self.mpi4py_installed:
            return self.comm.allgather(obj)
        return [obj, ]

    def allreduce(self, val, op: Op = SUM):
        """Alias of ``mpi4py.MPI.Comm.allreduce``"""
        if self.mpi4py_installed:
            return self.comm.allreduce(val, op)
        return val

    def Bcast(self, buf: BufSpec, root: int = 0):  # noqa: N802
        """Alias of ``mpi4py.MPI.Comm.Bcast``"""
        if self.mpi4py_installed:
            self.comm.Bcast(buf, root)

    def Scatter(self, sendbuf: BufSpec | None, recvbuf: InPlace | BufSpec,
                root: int = 0):
        if self.mpi4py_installed:
            self.comm.Scatter(sendbuf, recvbuf, root)
        elif recvbuf != IN_PLACE:
            recvbuf_ = recvbuf[0] if isinstance(recvbuf, tuple) else recvbuf
            sendbuf_ = sendbuf[0] if isinstance(sendbuf, tuple) else sendbuf
            recvbuf_[:] = sendbuf_

    def Scatterv(self, sendbuf: BufSpecV | None, recvbuf: InPlace | BufSpec,
                 root: int = 0):
        if self.mpi4py_installed:
            self.comm.Scatterv(sendbuf, recvbuf, root)
        elif recvbuf != IN_PLACE:
            recvbuf_ = recvbuf[0] if isinstance(recvbuf, tuple) else recvbuf
            sendbuf_ = sendbuf[0] if isinstance(sendbuf, tuple) else sendbuf
            recvbuf_[:] = sendbuf_

    def Gather(self, sendbuf: InPlace | BufSpec, recvbuf: BufSpec | None,
               root: int = 0):
        if self.mpi4py_installed:
            self.comm.Gather(sendbuf, recvbuf, root)
        elif sendbuf != IN_PLACE:
            recvbuf_ = recvbuf[0] if isinstance(recvbuf, tuple) else recvbuf
            sendbuf_ = sendbuf[0] if isinstance(sendbuf, tuple) else sendbuf
            recvbuf_[:] = sendbuf_

    def Gatherv(self, sendbuf: InPlace | BufSpec, recvbuf: BufSpecV | None,
                root: int = 0):
        if self.mpi4py_installed:
            self.comm.Gatherv(sendbuf, recvbuf, root)
        elif sendbuf != IN_PLACE:
            recvbuf_ = recvbuf[0] if isinstance(recvbuf, tuple) else recvbuf
            sendbuf_ = sendbuf[0] if isinstance(sendbuf, tuple) else sendbuf
            recvbuf_[:] = sendbuf_

    def Allgather(self, sendbuf: InPlace | BufSpec,
                  recvbuf: BufSpec):  # noqa: N802
        """Alias of `mpi4py.MPI.Comm.Allgather`"""
        if self.mpi4py_installed:
            self.comm.Allgather(sendbuf, recvbuf)
        elif sendbuf != IN_PLACE:
            recvbuf[:] = sendbuf

    def Allgatherv(self, sendbuf: InPlace | BufSpec,
                  recvbuf: BufSpecV):  # noqa: N802
        """Alias of `mpi4py.MPI.Comm.Allgatherv`"""
        if self.mpi4py_installed:
            self.comm.Allgatherv(sendbuf, recvbuf)
        elif sendbuf != IN_PLACE:
            recvbuf[0][:] = sendbuf

    def Allreduce(self, sendbuf: InPlace | BufSpec,
                  recvbuf: BufSpec, op: Op = SUM):  # noqa: N802
        """Alias of `mpi4py.MPI.Comm.Allreduce`"""
        if self.mpi4py_installed:
            self.comm.Allreduce(sendbuf, recvbuf, op=op)


def split_comm_pwgrp(comm: QTMComm, pwgrp_size: int = 1):
    if not isinstance(comm, QTMComm):
        raise TypeError(f"'comm' must be a '{QTMComm}' instance. "
                        f"got type {type(comm)}")
    if comm.is_null:
        raise ValueError(f"'comm' must not be a null communicator.")
    comm_size, comm_rank = comm.size, comm.rank

    if not isinstance(pwgrp_size, int) or pwgrp_size <= 0:
        raise ValueError("'pwgrp_size' must be a positive integer. "
                         f"got {pwgrp_size} (type {type(pwgrp_size)}).")

    if comm_size % pwgrp_size != 0:
        raise ValueError("'pwgrp_size' must evenly divide 'comm's "
                         f"{comm_size} processes, but pwgrp_size = {pwgrp_size} "
                         f"is not a factor of comm.size = {comm_size}")

    color = comm_rank // pwgrp_size
    key = comm_rank % pwgrp_size

    pwgrp_comm = None
    if pwgrp_size != 1:
        pwgrp_comm = comm.Split(color, key)
    intercomm = comm.Split(key, color)

    return pwgrp_comm, intercomm
