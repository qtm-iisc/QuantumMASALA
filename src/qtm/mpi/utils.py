# from __future__ import annotations
from typing import Optional, Union, Sequence
__all__ = ['scatter_range', 'scatter_len']

import numpy as np


def scatter_len(len_: int, grp_size: int,
                grp_rank: Optional[int] = None) -> Union[int, Sequence[int]]:
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


def scatter_range(r: range, grp_size: int, grp_rank: Optional[int] = None,
                  round_robin: bool = False) -> Union[range, Sequence[range]]:
    if not isinstance(r, range):
        raise TypeError("'r' must be a 'range' instance. "
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
