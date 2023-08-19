from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence
__all__ = ['scatter_slice', 'scatter_len']

import numpy as np


def scatter_len(len_: int, grp_size: int,
                grp_rank: int | None = None) -> Sequence[int] | int:
    assert isinstance(len_, int)
    assert len_ >= 0
    assert isinstance(grp_size, int)
    assert grp_size > 0
    if grp_rank is not None:
        assert isinstance(grp_rank, int)
        assert 0 <= grp_rank < grp_size
    else:
        grp_rank = np.arange(grp_size, dtype='i8')

    return len_ // grp_size + (grp_rank < (len_ % grp_size))


def scatter_slice(len_: int, grp_size: int,
                  grp_rank: int | None = None) -> slice | Sequence[slice]:
    assert isinstance(len_, int)
    assert len_ >= 0
    assert isinstance(grp_size, int)
    assert grp_size > 0
    if grp_rank is not None:
        assert isinstance(grp_rank, int)
        assert 0 <= grp_rank < grp_size
    else:
        grp_rank = np.arange(grp_size, dtype='i8')

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
