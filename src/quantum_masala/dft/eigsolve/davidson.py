__all__ = ['solver']

import numpy as np

from quantum_masala import config
from quantum_masala.dft.ksham import KSHam


def solver(ham: KSHam,
           diago_thr: float, evc_gk: np.ndarray,
           vbare_g0: float):
    # Choosing backend
    if config.use_gpu:
        import cupy as xp
    else:
        import numpy as xp
        from scipy.linalg import eigh

    # Setting up preconditioner
    ham_diag = ham.ke_gk + vbare_g0 \
        + np.sum(np.diag(ham.dij).reshape(-1, 1)
                 * (ham.l_vkb_H.T * ham.l_vkb), axis=0)

    def g_psi(psi_: xp.ndarray, l_evl_: xp.ndarray):
        nonlocal ham_diag
        scala = 2
        x = scala * (ham_diag.reshape(1, -1) - l_evl_.reshape(-1, 1))
        denom = 0.5 * (1 + x + np.sqrt(1 + (x - 1) * (x - 1))) / scala
        psi_ /= denom

    # Intra k-group Communicator for band parallelization
    kgrp_intracomm = config.pwcomm.kgrp_intracomm

    numeig, numgk = evc_gk.shape

    # Buffer for solution
    evc = xp.empty((numeig, numgk), dtype=xp.complex128)
    evl = xp.empty(numeig, dtype=xp.complex128)
    unconv_flag = xp.ones(numeig, dtype=xp.bool)
    nunconv = numeig

    # Setting initial guess
    evc[:] = evc_gk
    evc[:] /= xp.linalg.norm(evc, axis=1, keepdims=True)
    kgrp_intracomm.Bcast(evc)

    # Initializing Workspace
    ndim_max = config.davidson_numwork * numeig
    psi = xp.empty((ndim_max, numgk), dtype=xp.complex128)
    hpsi = xp.empty((ndim_max, numgk), dtype=xp.complex128)
    ham_red = xp.empty((ndim_max, ndim_max), dtype=xp.complex128)
    ovl_red = xp.empty((ndim_max, ndim_max), dtype=xp.complex128)
    evc_red = xp.empty((numeig, ndim_max), dtype=xp.complex128, order='F')
    evl_red = xp.empty(numeig, dtype=xp.complex128)

    ndim = numeig
    psi[:ndim] = evc

    numhpsi = 0
    def compute_hpsi(istart: int, istop: int):
        nonlocal numhpsi
        numhpsi += max(istop - istart, 0)
        sl = kgrp_intracomm.psi_scatter_slice(istart, istop)
        if sl.stop > sl.start:
            ham.h_psi(psi[sl], hpsi[sl])
        kgrp_intracomm.barrier()
        kgrp_intracomm.psi_Allgather_inplace(hpsi[istart:istop])

    def move_unconv():
        evc_red[:nunconv] = evc_red[:numeig][unconv_flag]
        evl_red[:nunconv] = evl_red[:numeig][unconv_flag]

    def expand_psi():
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

    def solve_red():
        ham_red_ = ham_red[:ndim, :ndim]
        ovl_red_ = ovl_red[:ndim, :ndim]
        evc_red_ = evc_red[:, :ndim].T

        sl_bgrp = kgrp_intracomm.psi_scatter_slice(ndim - nunconv, ndim)
        ham_red_[sl_bgrp] = psi[sl_bgrp].conj() @ hpsi[:ndim].T
        ovl_red_[sl_bgrp] = psi[sl_bgrp].conj() @ psi[:ndim].T
        kgrp_intracomm.psi_Allgather_inplace(ham_red[ndim-nunconv:ndim])
        kgrp_intracomm.psi_Allgather_inplace(ovl_red[ndim-nunconv:ndim])

        if kgrp_intracomm.rank == 0:
            # Generalized Eigenvalue Problem Ax = eBx
            evl_red[:], evc_red_[:] = eigh(ham_red_, ovl_red_,
                                           subset_by_index=[0, numeig-1])
            # evl_red[:], evc_red_[:] = lobpcg(ham_red_, evc_red_, ovl_red_,
            #                                  largest=False)
            # A, B = ham_red_, ovl_red_
            # L = xp.linalg.cholesky(B)
            # L_inv = xp.linalg.inv(L)
            # A_ = L_inv @ A @ L_inv.conj().T
            #
            # A_evl, A_evc = xp.linalg.eigh(A_, 'L')
            #
            # evl_red[:numeig] = A_evl[:numeig]
            # evc_red[:, :ndim] = xp.linalg.solve(L.conj().T, A_evc[:, :numeig]).T

        kgrp_intracomm.Bcast(evl_red)
        kgrp_intracomm.Bcast(evc_red[:, :ndim])

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
            ham_red[(range(numeig), range(numeig))] = evl
            ovl_red[(range(numeig), range(numeig))] = 1
            evc_red[(range(numeig), range(numeig))] = 1

    return evl.real, evc, {'numiter': idxiter, 'numhpsi': numhpsi}
