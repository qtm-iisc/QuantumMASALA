"""Eigensolver implementation based on `scipy.sparse.linalg.eigsh`

This implementation is purely for demonstration.
"""
from __future__ import annotations
__all__ = ['solve']
from scipy.sparse.linalg import LinearOperator, eigsh

from qtm.containers import WavefunGType
from qtm.dft import KSWfn, KSHam, DFTCommMod


def solve(dftcomm: DFTCommMod, ksham: KSHam, kswfn: KSWfn,
          diago_thr: float, *args, **kwargs) -> tuple[KSWfn, int]:
    assert isinstance(dftcomm, DFTCommMod)
    assert dftcomm.kgrp_intra.size == 1, \
        "band and PW distribution not possible with 'scipy' routines"

    kgrp_intra = dftcomm.kgrp_intra

    with kgrp_intra as comm:
        assert isinstance(ksham, KSHam)
        assert isinstance(kswfn, KSWfn)
        assert ksham.gkspc is kswfn.gkspc
        assert isinstance(diago_thr, float)
        assert diago_thr > 0

        gkspc = ksham.gkspc
        WavefunG = type(kswfn.evc_gk)
        numbnd = kswfn.numbnd
        evl = kswfn.evl
        evc_gk = kswfn.evc_gk.data.T
        n_hpsi = 0

        def ksham_matvec(psi):
            nonlocal n_hpsi
            n_hpsi += 1
            psi = WavefunG(psi)
            hpsi = WavefunG.empty(psi.shape)
            ksham.h_psi(psi, hpsi)
            return hpsi.data

        ksham_lo = LinearOperator((gkspc.size_g, gkspc.size_g),
                                  matvec=ksham_matvec,
                                  matmat=ksham_matvec,
                                  dtype='c16')

        evl[:], evc_gk[:] = eigsh(ksham_lo, numbnd, which='SA')

    return kswfn, n_hpsi
