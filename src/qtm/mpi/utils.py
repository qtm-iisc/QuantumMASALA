from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
__all__ = ["scatter_slice", "scatter_len"]

import numpy as np


def scatter_len(
    len_: int, grp_size: int, grp_rank: int | None = None
) -> Sequence[int] | int:
    assert isinstance(len_, int)
    assert len_ >= 0
    assert isinstance(grp_size, int)
    assert grp_size > 0
    if grp_rank is not None:
        assert isinstance(grp_rank, int)
        assert 0 <= grp_rank < grp_size
    else:
        grp_rank = np.arange(grp_size, dtype="i8")

    return len_ // grp_size + (grp_rank < (len_ % grp_size))


def scatter_slice(
    len_: int, grp_size: int, grp_rank: int | None = None
) -> slice | Sequence[slice]:
    assert isinstance(len_, int)
    assert len_ >= 0
    assert isinstance(grp_size, int)
    assert grp_size > 0

    all_ranks = grp_rank is None
    if all_ranks:
        # One slice per rank, computed in a single vectorized pass.
        ranks = np.arange(grp_size, dtype="i8")
    else:
        assert isinstance(grp_rank, int)
        assert 0 <= grp_rank < grp_size
        ranks = grp_rank

    start = (len_ // grp_size) * ranks + np.minimum(ranks, len_ % grp_size)
    stop = start + ((len_ // grp_size) + (ranks < len_ % grp_size))

    if not all_ranks:
        return slice(int(start), int(stop))
    return tuple(slice(int(start[r]), int(stop[r])) for r in range(grp_size))
