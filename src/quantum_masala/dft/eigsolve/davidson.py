__all__ = ['solver']

import numpy as np

from quantum_masala import config, pw_counter
from quantum_masala.dft.ksham import KSHam

def solver(ham: KSHam,
           diago_thr: float, evc_gk: np.ndarray,
           vbare_g0: float):
    pw_counter.start_clock('david')
    # Choosing backend
    if config.use_gpu:
        import cupy as xp
    else:
        import numpy as xp
        from scipy.linalg import eigh
    # Setting up preconditioner
    ham_diag = ham.ke_gk + vbare_g0 + ham.vnl_diag

    def g_psi(psi_: xp.ndarray, l_evl_: xp.ndarray):
        pw_counter.start_clock('david:g_psi')
        nonlocal ham_diag
        scala = 2
        for ipsi, evl_ in enumerate(l_evl_):
            x = scala * (ham_diag - evl_)
            denom = 0.5 * (1 + x + np.sqrt(1 + (x - 1) ** 2)) / scala
            psi_[ipsi] /= denom
        pw_counter.stop_clock('david:g_psi')

    # Intra k-group Communicator for band parallelization
    kgrp_intracomm = config.pwcomm.kgrp_intracomm

    numeig, numgk = evc_gk.shape

    # Buffer for solution and initializing it
    evc = xp.array(evc_gk, dtype=xp.complex128)
    evl = xp.empty(numeig, dtype=xp.complex128)
    unconv_flag = xp.ones(numeig, dtype=xp.bool)
    nunconv = numeig

    evc[:] /= xp.linalg.norm(evc, axis=1, keepdims=True)
    kgrp_intracomm.Bcast(evc)

    # Initializing Workspace
    ndim_max = config.davidson_numwork * numeig
    psi = xp.empty((ndim_max, numgk), dtype=xp.complex128)
    hpsi = xp.empty((ndim_max, numgk), dtype=xp.complex128)
    ham_red = xp.zeros((ndim_max, ndim_max), dtype=xp.complex128)
    ovl_red = xp.zeros((ndim_max, ndim_max), dtype=xp.complex128)
    evc_red = xp.empty((numeig, ndim_max), dtype=xp.complex128, order='F')
    evl_red = xp.empty(numeig, dtype=xp.complex128)

    ndim = numeig
    psi[:ndim] = evc

    def compute_hpsi(istart: int, istop: int):
        pw_counter.start_clock('david:h_psi')
        sl = kgrp_intracomm.psi_scatter_slice(istart, istop)
        if sl.stop > sl.start:
            ham.h_psi(psi[sl], hpsi[sl])
        kgrp_intracomm.barrier()
        kgrp_intracomm.psi_Allgather_inplace(hpsi[istart:istop])
        pw_counter.stop_clock('david:h_psi')

    def move_unconv():
        evc_red[:nunconv] = evc_red[:numeig][unconv_flag]
        evl_red[:nunconv] = evl_red[:numeig][unconv_flag]

    def expand_psi():
        pw_counter.start_clock('david:update')
        nonlocal ndim
        sl = slice(ndim, ndim + nunconv)
        sl_bgrp = kgrp_intracomm.psi_scatter_slice(0, ndim)
        psi[sl] = evc_red[:nunconv][(slice(None), sl_bgrp)] @ psi[sl_bgrp]
        hpsi[sl] = evc_red[:nunconv][(slice(None), sl_bgrp)] @ hpsi[sl_bgrp]
        kgrp_intracomm.Allreduce_sum_inplace(psi[sl])
        kgrp_intracomm.Allreduce_sum_inplace(hpsi[sl])

        psi[sl] *= -evl_red[:nunconv].reshape(-1, 1)
        psi[sl] += hpsi[sl]
        g_psi(psi[sl], evl_red[:nunconv])
        psi[sl] /= xp.linalg.norm(psi[sl], axis=1, keepdims=True)
        kgrp_intracomm.barrier()
        ndim += nunconv
        pw_counter.stop_clock('david:update')

    def solve_red():
        ham_red_ = ham_red[:ndim, :ndim]
        ovl_red_ = ovl_red[:ndim, :ndim]
        evc_red_ = evc_red[:, :ndim].T

        sl_bgrp = kgrp_intracomm.psi_scatter_slice(ndim-nunconv, ndim)
        ham_red_[sl_bgrp] = psi[sl_bgrp].conj() @ hpsi[:ndim].T
        ovl_red_[sl_bgrp] = psi[sl_bgrp].conj() @ psi[:ndim].T
        kgrp_intracomm.psi_Allgather_inplace(ham_red[0:ndim])
        kgrp_intracomm.psi_Allgather_inplace(ovl_red[0:ndim])

        pw_counter.start_clock('david:eigh')
        if kgrp_intracomm.rank == 0:
            if config.use_gpu:
                # Generalized Eigenvalue Problem Ax = eBx
                A, B = ham_red_, ovl_red_
                L = xp.linalg.cholesky(B)
                L_inv = xp.linalg.inv(L)
                A_ = L_inv @ A @ L_inv.conj().T

                A_evl, A_evc = xp.linalg.eigh(A_, 'L')

                evl_red[:numeig] = A_evl[:numeig]
                evc_red[:, :ndim] = xp.linalg.solve(L.conj().T, A_evc[:, :numeig]).T
            else:
                evl_red[:], evc_red_[:] = eigh(
                    ham_red_, ovl_red_, subset_by_index=[0, numeig-1],
                )
        kgrp_intracomm.Bcast(evl_red)
        kgrp_intracomm.Bcast(evc_red[:, :ndim])
        pw_counter.stop_clock('david:eigh')

    compute_hpsi(0, ndim)
    solve_red()
    evl[:] = evl_red[:numeig]

    idxiter = 0
    while idxiter < config.davidson_maxiter:
        idxiter += 1
        move_unconv()
        expand_psi()
        compute_hpsi(ndim - nunconv, ndim)
        solve_red()
        unconv_flag[:] = xp.abs(evl_red - evl) > diago_thr
        nunconv = xp.sum(unconv_flag)
        evl[:] = evl_red[:numeig]

        if nunconv == 0 or ndim + nunconv > ndim_max:
            evc[:] = evc_red[:numeig, :ndim] @ psi[:ndim]
            if nunconv == 0:
                break

            psi[:numeig] = evc[:]

            sl_bgrp = kgrp_intracomm.psi_scatter_slice(0, ndim)
            hpsi[:numeig] = evc_red[:numeig, sl_bgrp] @ hpsi[sl_bgrp]
            kgrp_intracomm.Allreduce_sum_inplace(hpsi[:numeig])

            ndim = numeig
            ham_red[:], ovl_red[:], evc_red[:, :] = 0, 0, 0
            xp.fill_diagonal(ham_red, evl)
            xp.fill_diagonal(ovl_red, 1)
            xp.fill_diagonal(evc_red, 1)

    pw_counter.stop_clock('david')
    if config.use_gpu:
        return xp.asnumpy(evl.real), xp.asnumpy(evc), idxiter
    else:
        return evl.real, evc, idxiter
