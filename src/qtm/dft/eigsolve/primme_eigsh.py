"""Eigensolver implementation based on `primme.eigsh`

This implementation is purely for demonstration.
"""

from __future__ import annotations
import numpy as np
from qtm.config import PRIMME_INSTALLED, NDArray
__all__ = ['solve']
from scipy.sparse.linalg import LinearOperator

from qtm.containers import WavefunGType
from qtm.dft import KSWfn, KSHam, DFTCommMod
from qtm.logger import qtmlogger


diago_method = 'david'

if diago_method == 'david':
    diago_method = 'PRIMME_GD_plusK'
elif diago_method == 'cg':
    diago_method = 'PRIMME_LOBPCG_OrthoBasis'
elif diago_method == 'ppcg':
    raise NotImplementedError("Diagonalization method not supported")
else:
    raise ValueError("'diagonalization' parameter not recognized")


def solve(dftcomm: DFTCommMod, ksham: KSHam, kswfn: KSWfn,
          diago_thr: float, vloc_g0: list[complex], numwork: int, maxiter: int) -> tuple[KSWfn, int]:
    if PRIMME_INSTALLED:
        import primme
    else:
        raise ImportError("Primme is not installed")
    assert isinstance(dftcomm, DFTCommMod)
    assert dftcomm.kgrp_intra.size == 1, \
        "band and PW distribution not possible with 'scipy' routines"

    kgrp_intra = dftcomm.kgrp_intra

    with kgrp_intra as comm:
        assert isinstance(ksham, KSHam)
        assert isinstance(kswfn, KSWfn)
        assert ksham.gkspc is kswfn.gkspc   # What??
        assert isinstance(diago_thr, float)
        assert diago_thr > 0

        gkspc = ksham.gkspc
        WavefunG = type(kswfn.evc_gk)
        numbnd = kswfn.numbnd
        evl = kswfn.evl
        evc_gk = kswfn.evc_gk.data.T
        n_hpsi = 0
        
        def ksham_matvec(psi_data:NDArray):
            nonlocal n_hpsi
            n_hpsi += 1
            psi = WavefunG.empty(psi_data.shape[1:])
            psi._data = psi_data.T
            hpsi = WavefunG.empty(psi_data.shape[1:])
            ksham.h_psi(psi, hpsi)
            return hpsi.data.T

        ksham_linoper = LinearOperator((gkspc.size_g, gkspc.size_g),
                                  matvec=ksham_matvec,
                                #   matmat=ksham_matvec,
                                  dtype='c16')

        
        ham_diag = ksham.ke_gk + ksham.vnl_diag
        ham_diag.data[..., :gkspc.size_g] += vloc_g0[0]
        
        def preconditioner(psi:NDArray, e_psi=None):
            nonlocal ham_diag
            if e_psi is None:
                e_psi = np.array(primme.get_eigsh_param('ShiftsForPreconditioner'))

            scala = 2.0
            x = scala * (ham_diag._data[:,None] - e_psi[None, :])
            denom = 0.5 * (1 + x + np.sqrt(1 + (x - 1) ** 2)) / scala
            psi /= denom * 2
            return psi

        Prec_oper = LinearOperator(shape=(kswfn.gkspc.size_g, kswfn.gkspc.size_g),
                dtype=np.complex128, matvec=preconditioner, rmatvec=preconditioner,
                matmat=preconditioner, rmatmat=preconditioner)


        # Alternative: TPA preconditioner

        # def tpa_prec(psi:NDArray):
        #     psi = psi.reshape(-1)
        #     E_k = 0.5
        #     x = (0.5 * ksham.gkspc.g_norm2) / E_k

        #     f = 27 + 18 * x + 12 * (x**2) + 8 * (x**3)
        #     prec = f / (f + 16 * (x**4))
        #     return psi * prec

        # def tpa_prec_mat(psi_mat:NDArray):
        #     num_psi = psi_mat.shape[1]
        #     E_k = 0.5 * np.ones(num_psi, dtype=float)
        #     x = (0.5 * ksham.gkspc.g_norm2.reshape(-1, 1)) / E_k

        #     f = 27 + 18 * x + 12 * (x**2) + 8 * (x**3)
        #     prec = f / (f + 16 * (x**4))
        #     return psi_mat * prec

        # Prec_oper = LinearOperator(shape=(kswfn.gkspc.size_g, kswfn.gkspc.size_g),
        #                         dtype=np.complex128, matvec=tpa_prec, rmatvec=tpa_prec,
        #                         matmat=tpa_prec_mat, rmatmat=tpa_prec_mat)
        

        evl[:], evc_gk[:], stats = primme.eigsh(   A=ksham_linoper, 
                                            k=numbnd, 
                                            v0=kswfn.evc_gk._data.T,
                                            which='SA', method=diago_method, 
                                            OPinv=Prec_oper,
                                            tol=diago_thr, aNorm=1, invBNorm=1,
                                            maxOuterIterations=maxiter,
                                            raise_for_unconverged=False, return_unconverged=True,
                                            maxBlockSize=numbnd, minRestartSize=numbnd,
                                            maxPrevRetain=numbnd,
                                            return_eigenvectors=True, return_stats=True, return_history=False,
                                        )
    return kswfn, n_hpsi

    