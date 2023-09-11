from __future__ import annotations
__all__ = ['compute_occ']

import numpy as np

from qtm.dft.kswfn import KSWfn
from qtm.dft.config import DFTCommMod
from .utils import check_args


def compute_occ(dftcomm: DFTCommMod, l_kswfn: list[list[KSWfn]], numel: int):
    with dftcomm.image_comm as comm:
        assert isinstance(numel, int)
        assert numel == comm.bcast(numel)
        assert numel % 2 == 0
        assert all(len(kswfn_k) == 1 for kswfn_k in l_kswfn)

    with dftcomm.image_comm as image_comm:
        numel = image_comm.bcast(numel)
        numbnd = l_kswfn[0][0].numbnd
        numfill = numel // 2
        for wfn_k in l_kswfn:
            assert len(wfn_k) == 1
            wfn_k[0].occ[:numfill] = 1
            wfn_k[0].occ[numfill:] = 0

        max_filled, min_empty = None, None
        with dftcomm.kroot_intra as comm:
            if comm.is_null:
                comm.skip_with_block()

            max_filled = comm.allreduce(max(
                np.amax(wfn_k[0].evl[:numfill]) for wfn_k in l_kswfn
            ), comm.MAX)
            if numfill < numbnd:
                min_empty = comm.allreduce(min(
                    np.amin(wfn_k[0].evl[numfill:]) for wfn_k in l_kswfn
                ), comm.MIN)
        max_filled = comm.bcast(max_filled)
        min_empty = comm.bcast(min_empty)

    return max_filled, min_empty
