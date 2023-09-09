from __future__ import annotations
__all__ = ['compute_occ']

import numpy as np

from qtm.dft.kswfn import KSWfn
from qtm.dft.config import DFTCommMod
from .utils import check_args


def compute_occ(dftcomm: DFTCommMod, l_kswfn: list[KSWfn], numel: int):
    check_args(dftcomm, l_kswfn, numel)

    with dftcomm.image_comm as image_comm:
        numel = image_comm.bcast(numel)
        assert isinstance(numel, int) and numel % 2 == 0

        numbnd = l_kswfn[0].numbnd
        numfill = numel // 2
        for wfn, in l_kswfn:
            wfn.occ[:, :numfill] = 1
            wfn.occ[:, numfill:] = 0

        max_filled, min_empty = None, None
        with dftcomm.kroot_intra as comm:
            max_filled = comm.allreduce(float(
                max(np.amax(wfn.evl[:, numfill]) for wfn in l_kswfn)
            ), comm.MAX)
            if numfill < numbnd:
                min_empty = comm.allreduce(float(
                    min(np.amin(wfn.evl[numfill:]) for wfn in l_kswfn)
                ), comm.MIN)
        max_filled = comm.bcast(max_filled)
        min_empty = comm.bcast(min_empty)

    return max_filled, min_empty
