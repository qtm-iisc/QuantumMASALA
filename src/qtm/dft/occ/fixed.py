__all__ = ['compute_occ']

import numpy as np

from qtm.dft.kswfn import KSWfn
from qtm.dft.comm_mod import DFTCommMod
from .utils import check_args


def compute_occ(dftcomm: DFTCommMod, l_wfn: list[KSWfn], numel: int):
    check_args(dftcomm, l_wfn, numel)

    with dftcomm.image_comm as image_comm:
        numel = image_comm.bcast(numel)
        assert isinstance(numel, int) and numel % 2 == 0

        numbnd = l_wfn[0].numbnd
        numfill = numel // 2
        for wfn in l_wfn:
            wfn.occ[:, :numfill] = 1
            wfn.occ[:, numfill:] = 0

        max_filled, min_empty = None, None
        with dftcomm.kroot_intra as comm:
            max_filled = comm.allreduce(float(
                max(np.amax(wfn.evl[:, numfill]) for wfn in l_wfn)
            ), comm.MAX)
            if numfill < numbnd:
                min_empty = comm.allreduce(float(
                    min(np.amin(wfn.evl[numfill:]) for wfn in l_wfn)
                ), comm.MIN)
        max_filled = comm.bcast(max_filled)
        min_empty = comm.bcast(min_empty)

    return max_filled, min_empty
