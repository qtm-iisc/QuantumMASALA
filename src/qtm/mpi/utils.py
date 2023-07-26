# from __future__ import annotations
from typing import Optional, Union, Sequence
__all__ = ['scatter_range', 'scatter_slice', 'scatter_len',
           'gen_subarray_dtypes', 'gen_vector_dtype']

import numpy as np


def scatter_len(len_: int, grp_size: int,
                grp_rank: Optional[int] = None) -> Union[Sequence[int], int]:
    if not isinstance(len_, int) or len_ < 0:
        raise ValueError("'len_' must be a non-negative integer. "
                         f"got {len_} (type {type(len_)})")
    if not isinstance(grp_size, int) or grp_size <= 0:
        raise ValueError("'grp_size' must be a positive integer. "
                         f"got {grp_size} (type {type(grp_size)})")
    if grp_rank is None:
        grp_rank = np.arange(grp_size, dtype='i8')
    elif not isinstance(grp_rank, int) or not 0 <= grp_rank < grp_size:
        raise ValueError("'grp_size' must be a non-negative integer that is"
                         f"less than 'grp_size'.\ngot grp_size = {grp_size}, "
                         f"grp_rank = {grp_rank} (type {type(grp_rank)})."
                         )

    return len_ // grp_size + (grp_rank < (len_ % grp_size))


def scatter_slice(len_: int, grp_size: int,
                  grp_rank: Optional[int] = None) -> Union[slice, Sequence[slice]]:
    if not isinstance(len_, int) or len_ < 0:
        raise TypeError("'len_' must be a positive integer. "
                        f"got type {type(len_)}. ")
    if not isinstance(grp_size, int) or grp_size <= 0:
        raise ValueError("'grp_size' must be a positive integer. "
                         f"got {grp_size} (type {type(grp_size)})")
    elif grp_rank is not None:
        if not isinstance(grp_rank, int) or not 0 <= grp_rank < grp_size:
            raise ValueError("'grp_size' must be a non-negative integer that is"
                             f"less than 'grp_size'.\ngot grp_size = {grp_size}, "
                             f"grp_rank = {grp_rank} (type {type(grp_rank)})."
                             )
    all_ranks = False
    if grp_rank is None:
        grp_rank = np.arange(grp_size, dtype='i8')
        all_ranks = True

    start = (len_ // grp_size) * grp_rank + min(grp_rank, len_ % grp_size)
    stop = start + ((len_ // grp_size) + (grp_rank < len_ % grp_size))

    if not all_ranks:
        return slice(start, stop)
    return tuple(
        slice(start[grp_rank], stop[grp_rank])
        for grp_rank in range(grp_rank)
    )


def scatter_range(r: Union[range, int], grp_size: int, grp_rank: Optional[int] = None,
                  round_robin: bool = False) -> Union[range, Sequence[range]]:
    if not isinstance(r, (range, int)):
        raise TypeError("'r' must be either a 'range' instance or an integer. "
                        f"got type {type(r)}. ")
    if not isinstance(grp_size, int) or grp_size <= 0:
        raise ValueError("'grp_size' must be a positive integer. "
                         f"got {grp_size} (type {type(grp_size)})")
    elif grp_rank is not None:
        if not isinstance(grp_rank, int) or not 0 <= grp_rank < grp_size:
            raise ValueError("'grp_size' must be a non-negative integer that is"
                             f"less than 'grp_size'.\ngot grp_size = {grp_size}, "
                             f"grp_rank = {grp_rank} (type {type(grp_rank)})."
                             )
        if not isinstance(round_robin, bool):
            raise TypeError("'round_robin' must be a boolean. "
                            f"got type {type(round_robin)}")

    if isinstance(r, int):
        r = range(r)

    if round_robin:
        if isinstance(grp_rank, int):
            return range(r.start + grp_rank, r.stop, r.step * grp_size)
        return tuple(
            range(r.start + grp_rank, r.stop, r.step * grp_size)
            for grp_rank in range(grp_size)
        )
    else:
        all_ranks = False
        if grp_rank is None:
            grp_rank = np.arange(grp_size, dtype='i8')
            all_ranks = True

        len_r = len(r)
        start = r.start + (
            (len_r // grp_size) * grp_rank + min(grp_rank, len_r % grp_size)
        ) * r.step
        stop = start + (
            (len_r // grp_size) + (grp_rank < len_r % grp_size)
        ) * r.step

        if not all_ranks:
            return range(start, stop, r.step)
        return tuple(
            range(start[grp_rank], stop[grp_rank], r.step)
            for grp_rank in range(grp_rank)
        )


def gen_subarray_dtypes(shape: Sequence[int], axis: int, dtype_np: np.dtype, grp_size: int):
    from mpi4py.MPI import _typedict
    dtype_mpi = _typedict[dtype_np.char]

    subshape = list(shape)
    substarts = [0] * len(shape)
    subarray_dtypes = []
    for grp_rank in range(grp_size):
        sublen = scatter_len(shape[axis], grp_size, grp_rank)
        subshape[axis] = sublen
        subarray_dtypes.append(
            dtype_mpi.Create_subarray(shape, subshape, substarts).Commit()
        )
        substarts[axis] += sublen

    return subarray_dtypes


def gen_vector_dtype(shape: Sequence[int], axis: int, dtype_np: np.dtype):
    from mpi4py.MPI import _typedict
    dtype_mpi = _typedict[dtype_np.char]

    nblocks = np.prod(shape[:axis], dtype='i8')
    stridelen = np.prod(shape[axis:], dtype='i8')
    blocklen = stridelen // shape[axis]
    vectype = dtype_mpi.Create_vector(
        nblocks, blocklen, stridelen
    ).Commit()
    vectype = vectype.Create_resized(0, dtype_mpi.size).Commit()
    return vectype
