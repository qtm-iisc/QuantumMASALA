__all__ = ['compute_occ']

import numpy as np

from quantum_masala.core import Wavefun
from quantum_masala import config


def compute_occ(l_wfn: list[Wavefun], numel: int):
    # Check if weights of k-points add to one
    k_weight_sum = 0
    pwcomm = config.pwcomm
    if pwcomm.kgrp_rank == 0:
        k_weight_sum = pwcomm.kgrp_intercomm.allreduce_sum(
            sum(wfn.k_weight for wfn in l_wfn)
        )
    k_weight_sum = pwcomm.world_comm.bcast(k_weight_sum)
    if abs(k_weight_sum - 1) >= 1E-7:
        raise ValueError("sum of 'wfn.k_weight' across all Wavefun instances 'wfn' "
                         f"must equal 1. Got {k_weight_sum}")
    if not isinstance(numel, int) or numel <= 0 or numel % 2 != 0:
        raise ValueError("'numel' must be a positive even integer. "
                         f"Got a {type(numel)} with value {numel}")

    numbnd = l_wfn[0].numbnd
    numfill = numel // 2
    for wfn in l_wfn:
        wfn.occ[:, :numfill] = 1
        wfn.occ[:, numfill:] = 0

    max_filled, min_empty = None, None
    if pwcomm.kgrp_rank == 0:
        max_filled = pwcomm.kgrp_intercomm.allreduce_max(
            max(np.amax(wfn.evl[:, numfill - 1]) for wfn in l_wfn)
        )
        if numfill < numbnd:
            min_empty = pwcomm.kgrp_intercomm.allreduce_min(
                min(np.amin(wfn.evl[:, numfill]) for wfn in l_wfn)
            )
    max_filled = pwcomm.world_comm.bcast(max_filled)
    if numfill < numbnd:
        min_empty = pwcomm.world_comm.bcast(min_empty)

    return max_filled, min_empty
