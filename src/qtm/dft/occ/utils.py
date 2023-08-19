from __future__ import annotations
__all__ = ['check_args']

from qtm.dft.kswfn import KSWfn
from qtm.dft.comm_mod import DFTCommMod

EPS = 1E-7


def check_args(dftcomm: DFTCommMod, l_wfn: list[KSWfn], numel: float | int):
    # Checking type of 'dftcomm'
    assert isinstance(dftcomm, DFTCommMod)
    with dftcomm.image_comm as image_comm:
        for wfn in l_wfn:
            # Checking type of all elements in 'l_wfn'
            assert isinstance(wfn, KSWfn)
            # Ensuring 'numbnd' matches across all elements in 'l_wfn'
            assert wfn.numbnd == l_wfn[0].numbnd
            # Ensuring attributes 'acc' and 'evl' of each 'wfn' in 'l_wfn'
            # match across all processes
            with dftcomm.kgrp_intra as comm:
                comm.Bcast(wfn.occ)
                comm.Bcast(wfn.evl)

        # Checking type of numel
        assert isinstance(numel, (float, int)) and numel > 0
        # Checking if 'numel' is identical across all processes
        assert (numel - image_comm.bcast(numel)) < EPS
