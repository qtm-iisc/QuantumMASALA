"""Davidson's diagonalization scheme method with overlap

Based on the implementation in QuantumESPRESSO
"""
from __future__ import annotations
__all__ = ['solve']
from numbers import Number
import numpy as np
from scipy.linalg import eigh

from qtm.containers import WavefunGType
from qtm.dft import KSWfn, KSHam, DFTCommMod

from qtm.logger import qtmlogger
from qtm.config import NDArray, CUPY_INSTALLED


@qtmlogger.time('davidson')
def solve(dftcomm: DFTCommMod, ksham: KSHam, kswfn: KSWfn, diago_thr: float,
          vloc_g0: list[complex], numwork: int, maxiter: int) -> tuple[KSWfn, int]:
    assert isinstance(dftcomm, DFTCommMod)

    kgrp_intra = dftcomm.kgrp_intra
    bgrp_inter = dftcomm.pwgrp_inter_kgrp

    with kgrp_intra as comm:
        assert isinstance(ksham, KSHam)
        assert isinstance(kswfn, KSWfn)
        assert ksham.gkspc is kswfn.gkspc

        diago_thr = comm.bcast(diago_thr)
        assert isinstance(diago_thr, float)
        assert diago_thr > 0

        vloc_g0 = comm.bcast(vloc_g0)
        # assert isinstance(vloc_g0, Number), print(type(vloc_g0), vloc_g0)

        numwork = comm.bcast(numwork)
        assert isinstance(numwork, int) and numwork > 1

        maxiter = comm.bcast(maxiter)
        assert isinstance(maxiter, int) and maxiter > 1

    WavefunG: type[WavefunGType] = type(kswfn.evc_gk)
    gkspc = WavefunG.gkspc
    basis_size = WavefunG.basis_size
    is_noncolin = kswfn.is_noncolin
    numeig = kswfn.numbnd
    with kgrp_intra as comm:
        comm.Bcast(kswfn.evl)
    with bgrp_inter as comm:
        comm.Bcast(kswfn.evc_gk.data)

    # Approximate Inverse of Hamiltonian
    # Only the diagonal part of the Ham is considered and 'approximately'
    # inverted so that the small terms will not be transformed to large
    # quantities
    ham_diag = ksham.ke_gk + ksham.vnl_diag
    ham_diag.data[..., :gkspc.size_g] += vloc_g0[0]
    if is_noncolin:
        ham_diag.data[..., gkspc.size_g:] += vloc_g0[1]

    @qtmlogger.time('davidson:apply_g_psi')
    def apply_g_psi(l_wfn: WavefunGType, l_evl: NDArray):
        scala = 2.0
        for wfn_, evl_ in zip(l_wfn, l_evl):
            x = scala * (ham_diag - evl_)
            denom = 0.5 * (1 + x + np.sqrt(1 + (x - 1) ** 2)) / scala
            wfn_ /= denom*2

    evc = kswfn.evc_gk
    evl = gkspc.allocate_array(numeig)
    unconv_flag = np.ones(numeig, dtype='bool', like=evl)
    n_unconv = numeig

    ndim_max = numwork * numeig
    psi = WavefunG.empty(ndim_max)
    hpsi = WavefunG.empty(ndim_max)
    ham_red = np.zeros((ndim_max, ndim_max), dtype='c16', like=evl)
    ovl_red = np.zeros((ndim_max, ndim_max), dtype='c16', like=evl)
    evc_red = np.zeros((numeig, ndim_max), dtype='c16', like=evl)
    evl_red = np.zeros(numeig, dtype='c16', like=evl)

    ndim = numeig
    psi[:ndim] = evc
    psi[:ndim].normalize()

    # Function for scattering a given contiguous slice across band groups
    def scatter_slice(sl: slice, grpsize: int, grprank: int):
        sl_stop = sl.stop
        sl_start = sl.start if sl.start is not None else 0
        if sl_start >= sl_stop:
            return slice(sl_start, sl_stop)
        len_ = sl_stop - sl_start
        sca_start = sl_start \
            + grprank * (len_ // grpsize) \
            + min(grprank, len_ % grpsize)
        sca_stop = sca_start \
            + (len_ // grpsize) \
            + (grprank < len_ % grpsize)
        return slice(sca_start, sca_stop)

    def gen_bufspec(len_, grpsize):
        grprank = np.arange(grpsize, dtype='i8', like=evl)
        return (len_ // grpsize) + (grprank < len_ % grpsize)

    @qtmlogger.time('davidson:compute_hpsi')
    def compute_hpsi(istart, istop):
        with bgrp_inter as comm:
            sl_psi = slice(istart, istop)
            sl_bgrp = scatter_slice(sl_psi, comm.size, comm.rank)
            if sl_bgrp.stop > sl_bgrp.start:
                ksham.h_psi(psi[sl_bgrp], hpsi[sl_bgrp])
            comm.Barrier()
            bufspec = gen_bufspec(istop - istart, comm.size) * basis_size
            comm.Allgatherv(bgrp_inter.IN_PLACE, (
                hpsi[sl_psi].data, bufspec
            ))

    # Grouping the unconverged eigenvectors of the reduced hamiltonian together
    # For batched operations using matmul (GEMM)
    def move_unconv():
        evc_red[:n_unconv] = evc_red[unconv_flag]
        evl_red[:n_unconv] = evl_red[unconv_flag]

    # Since only the lower triangular matrix is used anyway
    # This is not required
    def hermitianize_mat(mat: NDArray, n_end: int):
        # Using the values in the lower left rectangular block
        # to fill the values in the top right conjugate
        ll_blk = mat[-n_end:, :-n_end]  # lower left (horizontal) block
        ru_blk = mat[:-n_end, -n_end:]  # upper right (vertical) block
        ru_blk[:] = ll_blk.T.conj()
        # Using the lower triangle of the lower right square matrix
        # To fill the values in the upper triangle of the same
        rl_blk = mat[-n_end:, -n_end:]  # right lower block
        i_tril = np.tril_indices_from(rl_blk, -1)
        i_triu = (i_tril[1], i_tril[0])
        rl_blk[i_triu] = rl_blk[i_tril].conj()
        # Making the diagonal of the lower right square matrix real
        i_diag = np.diag_indices_from(rl_blk)
        rl_blk[i_diag] = rl_blk[i_diag].real

    # New Wavefunctions are constructed using the unconverged eigenvectors
    # of the reduced hamiltonian. The residual of the resulting wavefunctions
    # are used to expand the subspace. Preconditioning is applied to the
    # residuals before adding to the basis of the subspace
    @qtmlogger.time('davidson:expand_psi')
    def expand_psi():
        nonlocal ndim
        sl_curr = slice(ndim)
        sl_new = slice(ndim, ndim + n_unconv)
        with bgrp_inter as comm:
            sl_bgrp = scatter_slice(sl_curr, comm.size, comm.rank)

            np.matmul(evc_red[:n_unconv, sl_bgrp], psi[sl_bgrp], out=psi[sl_new])
            np.matmul(evc_red[:n_unconv, sl_bgrp], hpsi[sl_bgrp], out=hpsi[sl_new])
            comm.Allreduce(comm.IN_PLACE, psi[sl_new].data)
            comm.Allreduce(comm.IN_PLACE, hpsi[sl_new].data)

        psi[sl_new] *= -evl_red[:n_unconv].reshape((-1, 1))
        psi[sl_new] += hpsi[sl_new]

        apply_g_psi(psi[sl_new], evl_red[:n_unconv])
        psi[sl_new].normalize()
        compute_hpsi(ndim, ndim + n_unconv)
        ndim += n_unconv

    # The reduced hamiltonian matrix needs to be expanded with the newly added
    # basis wavefunctions. The previously computed regions of the reduced
    # hamiltonian is not computed again.
    def update_red():
        ham_red_ = ham_red[:ndim, :ndim]
        ovl_red_ = ovl_red[:ndim, :ndim]

        ham_red_[:] = psi[:ndim].vdot(hpsi[:ndim])
        ovl_red_[:] = psi[:ndim].vdot(psi[:ndim])

        sl_new = slice(ndim - n_unconv, ndim)
        with bgrp_inter as comm:
            sl_bgrp = scatter_slice(sl_new, comm.size, comm.rank)
            if sl_new.stop > sl_new.start:
                ham_red_[sl_bgrp] = psi[sl_bgrp].vdot(hpsi[:ndim])
                ovl_red_[sl_bgrp] = psi[sl_bgrp].vdot(psi[:ndim])
            comm.barrier()

            bufspec = gen_bufspec(n_unconv, comm.size) * ndim_max
            comm.Allgatherv(
                comm.IN_PLACE, (ham_red[sl_new], bufspec)
            )
            comm.Allgatherv(
                comm.IN_PLACE, (ovl_red[sl_new], bufspec)
            )

        # hermitianize_mat(ham_red_, n_unconv)
        # hermitianize_mat(ovl_red_, n_unconv)

    # The reduced hamiltonian is solved and the corresponding eigenpairs
    # are broadcasted to all processes in kgrp
    if WavefunG.ndarray is np.ndarray:
        def solve_red():
            ham_red_ = ham_red[:ndim, :ndim]
            ovl_red_ = ovl_red[:ndim, :ndim]
            evc_red_ = evc_red[:, :ndim].T

            with kgrp_intra as comm:
                if comm.rank == 0:
                    evl_red[:], evc_red_[:] = eigh(
                        ham_red_, ovl_red_, subset_by_index=[0, numeig - 1],
                        lower=True
                    )
                comm.Bcast(evc_red)
                comm.Bcast(evl_red)
    elif CUPY_INSTALLED:
        import cupy as cp
        if WavefunG.ndarray is not cp.ndarray:
            raise TypeError

        from cupyx import seterr
        seterr(linalg='raise')
        def solve_red():
            ham_red_ = ham_red[:ndim, :ndim]
            ovl_red_ = ovl_red[:ndim, :ndim]
            # print('solve', np.sum(np.isnan(ham_red_)), np.sum(np.isnan(ovl_red_)))
            # print(ovl_red_)

            with kgrp_intra as comm:
                if comm.rank == 0:
                    # Generalized Eigenvalue Problem Ax = eBx
                    A, B = ham_red_, ovl_red_
                    L = cp.linalg.cholesky(B)
                    # print(np.sum(np.isnan(L)))
                    L_inv = cp.linalg.inv(L)
                    # print(np.sum(np.isnan(L_inv)))
                    A = cp.tril(A, -1) + cp.tril(A, -1).T.conj() \
                        + cp.diag(cp.diag(A).real)
                    A_ = L_inv @ A @ L_inv.conj().T
                    # print(np.sum(np.isnan(A_)))
                    A_evl, A_evc = cp.linalg.eigh(A_, 'L')
                    evl_red[:numeig] = A_evl[:numeig]
                    evc_red[:, :ndim] = cp.linalg.solve(L.conj().T, A_evc[:, :numeig]).T
                comm.Bcast(evc_red)
                comm.Bcast(evl_red)
            # print('solve', np.sum(np.isnan(ham_red_)), np.sum(np.isnan(ovl_red_)),
            #       np.sum(np.isnan(evl[:numeig])), np.sum(np.isnan(evc_red[:, :ndim])))
            # print(evl_red)
    else:
        raise TypeError('not supported')

    # Starting with input wavefunctions
    compute_hpsi(0, ndim)
    update_red()
    solve_red()
    evl[:] = evl_red[:numeig]

    idxiter = 0
    while idxiter < maxiter:
        # debug: print('|'*30, idxiter, '|'*30)
        idxiter += 1
        move_unconv()
        expand_psi()
        update_red()
        solve_red()

        # Criteria for checking convergence is the change in the eigenvalues
        # between consecutive iterations
        with kgrp_intra as comm:
            unconv_flag[:] = np.abs(evl_red - evl) > diago_thr
            comm.Bcast(unconv_flag)

        n_unconv = np.sum(unconv_flag)
        evl[:] = evl_red[:]
        # debug: print(ndim, n_unconv)
        # If the number of basis vectors are exceeding the maximum limit
        # (as per 'numwork'), the basis needs to be reset/restarted
        # keeping only the required wavefunctions
        if n_unconv == 0 or ndim + n_unconv > ndim_max:
            with bgrp_inter as comm:
                sl_bgrp = scatter_slice(slice(ndim), comm.size, comm.rank)
                np.matmul(evc_red[:numeig, sl_bgrp], psi[sl_bgrp], out=evc[:])
                comm.Allreduce(comm.IN_PLACE, evc.data)
                if n_unconv == 0:
                    break
                # debug: print('restart')

                # WARNING: the new hpsi's are stored in psi temporarily
                # as the matmul output must not overlap with input arguments
                np.matmul(evc_red[:numeig, sl_bgrp], hpsi[sl_bgrp], out=psi[:numeig])
                comm.Allreduce(comm.IN_PLACE, psi[:numeig].data)
                hpsi[:numeig] = psi[:numeig]
                psi[:numeig] = evc[:]

            ndim = numeig
            ham_red[:], ovl_red[:], evc_red[:] = 0, 0, 0
            np.fill_diagonal(ham_red[:ndim, :ndim], evl.real)
            np.fill_diagonal(ovl_red[:ndim, :ndim], 1)
            np.fill_diagonal(evc_red[:ndim, :ndim], 1)

    kswfn.evc_gk[:] = evc

    kswfn.evl[:] = np.array(evl.real, like=kswfn.evl)
    # print(kswfn.evl)
    return kswfn, idxiter
