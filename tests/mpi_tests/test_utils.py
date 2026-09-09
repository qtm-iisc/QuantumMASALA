"""Simple, DFT-free unit tests for `qtm.mpi.utils.scatter_len`/
`scatter_slice`: pure index-partitioning functions, no communicator
involved, so fully testable in a single process.

Includes a regression test for a bug in `scatter_slice`'s `grp_rank=None`
("give me every rank's slice at once") branch: the sentinel `None` was
overwritten by a `np.arange(grp_size)` array before the code decided
whether to take that branch, so it never actually fired, and the fallback
arithmetic (a `min()` against an array, then a loop variable shadowing that
same array in `range(...)`) crashed outright. Every real caller in the
codebase (`qtm.kpts.KList.scatter`, `qtm.mpi.gspace`, `qtm.dft.scf`) always
passes an explicit integer rank, so this was dead-but-broken code, not yet
hit in production.
"""
import numpy as np
import pytest

from qtm.mpi.utils import scatter_len, scatter_slice


@pytest.mark.parametrize("len_", [0, 1, 7, 10, 23])
@pytest.mark.parametrize("grp_size", [1, 2, 3, 5])
def test_per_rank_slices_partition_the_full_range_exactly(len_, grp_size):
    covered = []
    for rank in range(grp_size):
        sl = scatter_slice(len_, grp_size, rank)
        covered.extend(range(sl.start, sl.stop))
    assert covered == list(range(len_))


@pytest.mark.parametrize("len_", [0, 1, 7, 10, 23])
@pytest.mark.parametrize("grp_size", [1, 2, 3, 5])
def test_scatter_len_matches_slice_lengths_and_sums_to_len(len_, grp_size):
    lens = scatter_len(len_, grp_size, None)
    assert np.sum(lens) == len_
    for rank in range(grp_size):
        sl = scatter_slice(len_, grp_size, rank)
        assert lens[rank] == sl.stop - sl.start
        assert scatter_len(len_, grp_size, rank) == lens[rank]


@pytest.mark.parametrize("len_", [0, 1, 7, 10, 23])
@pytest.mark.parametrize("grp_size", [1, 2, 3, 5])
def test_chunk_sizes_are_balanced_within_one(len_, grp_size):
    # No rank should get more than ceil(len_/grp_size) or fewer than
    # floor(len_/grp_size) elements.
    lens = [scatter_len(len_, grp_size, r) for r in range(grp_size)]
    assert max(lens) - min(lens) <= 1


def test_scatter_slice_rejects_out_of_range_rank():
    with pytest.raises(AssertionError):
        scatter_slice(10, 3, 3)  # rank must be < grp_size
    with pytest.raises(AssertionError):
        scatter_slice(10, 3, -1)


# ----- regression: grp_rank=None used to crash -------------------------------
@pytest.mark.parametrize("len_", [0, 1, 7, 10, 23])
@pytest.mark.parametrize("grp_size", [1, 2, 3, 5])
def test_scatter_slice_all_ranks_matches_individual_per_rank_calls(len_, grp_size):
    all_slices = scatter_slice(len_, grp_size, None)
    assert len(all_slices) == grp_size
    for rank in range(grp_size):
        assert all_slices[rank] == scatter_slice(len_, grp_size, rank)
