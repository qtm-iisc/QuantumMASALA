# from __future__ import annotations
from typing import Optional, Self, Sequence, Any, Union
from qtm.config import NDArray
__all__ = ['QTMComm', 'QTMIntracomm',
           'BufSpec', 'BufSpecV']

from types import MethodType
from qtm.logger import warn

from qtm import qtmconfig
if qtmconfig.mpi4py_installed:
    from mpi4py.MPI import COMM_WORLD, COMM_NULL, Intracomm, Group, Op
    from mpi4py.MPI import IN_PLACE, SUM, PROD, MIN, MAX, IDENT, SIMILAR

    WORLD_SIZE, WORLD_RANK = COMM_WORLD.Get_size(), COMM_WORLD.Get_rank()
    InPlace = type(IN_PLACE)
else:
    COMM_WORLD = None
    Intracomm, Op, InPlace = 'Intracomm', 'Op', 'InPlace'

    WORLD_SIZE, WORLD_RANK = 1, 0
    IN_PLACE, SUM, PROD, MIN, MAX = 'IN_PLACE', 'SUM', 'PROD', 'MIN', 'MAX'

BufSpec = NDArray
BufSpecV = tuple[NDArray, Sequence[int]]


class QTMComm:

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
    BUFFER_ATTR_NAME: str = '_data'
    """Attribute name in QTM's data containers that points to the underlying
    array/buffer"""

    def __init__(self, comm: Optional[Intracomm],
                 parent_comm: Optional[Self] = None,
                 sync_with_parent: bool = True,
                 id_: Any = None):
        # Getting installation status of mpi4py
        mpi4py_installed = qtmconfig.mpi4py_installed

        # Validating 'comm'
        is_null = False
        if mpi4py_installed:
            if comm == COMM_NULL:
                is_null = True
            elif not isinstance(comm, Intracomm):
                raise TypeError("'comm' must be a MPI Intracommunicator. "
                                f"got {type(comm)}.")
        elif comm is not None:
            raise ValueError(
                "'comm' takes only 'None' when mpi4py is not installed/enabled. "
                f"got {comm} (type {type(comm)}).")

        self.is_null: bool = is_null
        """True if `comm` is `mpi4py.MPI.COMM_NULL`."""

        self.comm: Optional[Intracomm] = comm
        """MPI Intracommunicator. Will be `None` if `qtm.qtmconfig.mpi4py_installed`
        is `False`.
        """

        size, rank = 1, 0
        if self.is_null:
            size = 0
        elif self.comm is not None:
            size, rank = self.comm.Get_size(), self.comm.Get_rank()
        self.size: int = size
        """Size of the group associated to `comm`.
        """
        self.rank: int = rank
        """Rank of the process in the group associated to `comm`.
        """

        if parent_comm is not None:
            if isinstance(parent_comm, Intracomm):
                parent_comm = QTMComm(parent_comm, None)
            if not isinstance(parent_comm, QTMComm):
                raise TypeError(
                    "If not 'None, 'parent_comm' must be an instance of either "
                    "'QTMComm' or 'mpi4py.MPI.Intracomm'. "
                    f"got type {type(parent_comm)}"
                )
            if parent_comm.is_null:
                warn("'parent_comm' is a null communicator. Setting it"
                     "to 'None'.")
                parent_comm = None

        if mpi4py_installed and not self.is_null and parent_comm is not None:
            comm_group = self.comm.Get_group()
            grp_intersection = Group.Intersection(
                comm_group, parent_comm.Get_group())
            if Group.Compare(comm_group, grp_intersection) not in [IDENT, SIMILAR]:
                raise ValueError("'comm' is not a subgroup of group associated to"
                                 "'parent_comm'.")
        self.parent_comm: Optional[QTMComm] = parent_comm
        """Parent Communicator; used to synchronize subgroups (of which this
        instance is a part of) when exiting context."""

        if not isinstance(sync_with_parent, bool):
            raise ValueError("'sync_with_parent' must be a boolean. "
                             f"got {type(sync_with_parent)}")
        if self.parent_comm is None:
            sync_with_parent = False
        self.sync_with_parent: bool = sync_with_parent
        """If True, call ``parent_comm.barrier()`` when exiting context."""

        self.id_: Any = id_
        """Attribute for labelling the group. When using `Split`, by default
        this will be the `color` argument.
        """

    def __getattribute__(self, item):
        """Overloaded to disable all methods if instance is a
        null communicator i.e `is_null` is True"""
        out = super().__getattribute__(item)
        if isinstance(out, MethodType) and self.is_null:
            raise AttributeError("'CommMod' instance is a null communicator. "
                                 "All methods except 'barrier' are disabled.")
        return out

    def __enter__(self) -> Self:
        """When used as a context manager, ``MPI_Barrier`` is called on entry
        to get all processes in the group to sync."""
        if not self.is_null:
            self.Barrier()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """When exiting the context, all processes in group will be
        synchronized with ``MPI_Barrier`` call. This is disabled if
        `sync_with_parent` is set to `False`."""
        if isinstance(exc_val, Exception):
            raise Exception(
                f"Process # {self.world_rank} / {self.world_size} has encountered "
                f"an exception. Please refer above for further info."
            ) from exc_val

        if not self.is_null:
            self.Barrier()
        if self.sync_with_parent:
            self.parent_comm.Barrier()

    def Barrier(self) -> None:  # noqa: N802
        if self.comm is not None:
            self.comm.Barrier()

    def Get_group(self) -> Optional[Group]:  # noqa: N802
        """Alias of ``self.comm.Get_group()``"""
        if self.comm is None:
            return None
        return self.comm.Get_group()

    def Incl(self, ranks: Sequence[int], sync_with_parent: bool = True) -> Self:  # noqa: N802
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
        if self.comm is None:
            if list(ranks) != [0, ]:
                raise ValueError("communicator has only one process")
            return self
        comm_out = Intracomm(self.comm.Create_group(
            self.comm.Get_group().Incl(ranks)
        ))
        return QTMComm(comm_out, self, sync_with_parent)

    def Split(self, color: int, key: int, sync_with_parent: bool = True,
              id_: Optional[Any] = None) -> Self:  # noqa: N802
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
        sync_with_parent : bool, default=True
            If True, when used in a `with` block, calls `parent_comm.Barrier()`
            so that all processes in the parent group will exit the `with`
            code block at the same time
        id_ : Optional[Any],default=color
            If specified, will be passed to `QTMComm` instance creation,
            else, `color` value will be used in place.
        """
        comm_split = Intracomm(
            self.comm.Split(color, key)
        )
        if id_ is None:
            id_ = color
        return QTMComm(comm_split, self, sync_with_parent, id_)

    def _check_op(self, op: Op) -> None:
        """Checks if input MPI operator is supported by this module"""
        l_op = [self.SUM, self.PROD, self.MIN, self.MAX]
        if op not in l_op:
            raise NotImplementedError("'op' is not supported/recognized. "
                                      f"Supported values: {l_op}")

    def bcast(self, obj):
        """Alias of ``mpi4py.MPI.Comm.bcast``"""
        if self.comm is None:
            return obj
        return self.comm.bcast(obj)

    def allgather(self, obj):
        """Alias of ``mpi4py.MPI.Comm.allgather``"""
        if self.comm is None:
            return [obj, ]
        return self.comm.allgather(obj)

    def allreduce(self, val, op: Op = SUM):
        """Alias of ``mpi4py.MPI.Comm.allreduce``"""
        self._check_op(op)
        if self.comm is None:
            return val
        return self.comm.allreduce(val, op)

    def Bcast(self, buf: BufSpec):  # noqa: N802
        """Alias of ``mpi4py.MPI.Comm.Bcast``"""
        if self.comm is not None:
            self.comm.Bcast(buf)
        return buf

    def Allgather(self, sendbuf: Union[InPlace, BufSpec],
                  recvbuf: BufSpec):  # noqa: N802
        """Alias of `mpi4py.MPI.Comm.Allgather`"""
        self.comm.Allgather(sendbuf, recvbuf)

    def Allgatherv(self, sendbuf: Union[InPlace, BufSpecV],
                  recvbuf: BufSpecV):  # noqa: N802
        """Alias of `mpi4py.MPI.Comm.Allgatherv`"""
        if self.comm is not None:
            self.comm.Allgather(sendbuf, recvbuf)

    def Allreduce(self, sendbuf: Union[InPlace, BufSpec],
                  recvbuf: BufSpec, op: Op = SUM):  # noqa: N802
        """Alias of `mpi4py.MPI.Comm.Allreduce`"""
        self._check_op(op)
        if self.comm is not None:
            self.comm.Allreduce(sendbuf, recvbuf, op=op)


class QTMIntracomm(QTMComm):

    def __init__(self, comm: QTMComm, pwgrp_size: int):
        if not isinstance(comm, QTMComm):
            raise TypeError("'comm' must be a 'QTMComm' instance. "
                            f"got '{type(comm)}' instance")
        if not isinstance(pwgrp_size, int) or pwgrp_size <= 0:
            raise TypeError("'pwgrp_size' must be a positive integer. "
                            f"got {pwgrp_size} (type {type(pwgrp_size)}")
        if comm.size % pwgrp_size != 0:
            raise ValueError("'pwgrp_size' must divide processes in 'comm' into"
                             "evenly sized subgroups. "
                             f"got comm.size = {comm.size}, pwgrp_size={pwgrp_size}")

        color, key = comm.rank // pwgrp_size, comm.rank % pwgrp_size
        pwgrp_intracomm = comm.Split(color, key)
        pwgrp_intercomm = comm.Split()


