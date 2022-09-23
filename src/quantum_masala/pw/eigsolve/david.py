import numpy as np

from quantum_masala.core import WfnK, KgrpIntraComm
from ..ham import HamK


from quantum_masala.config import CUPY_INSTALLED
# if CUPY_INSTALLED:
#     from pypwscf.gpu.ham import HamKGPU


MAXITER = 20


def david_solve(wfn: WfnK, ham: HamK, idxspin: int,
                diago_thr: float, vbare_g0: float, numwork: int = 4
):
    # Choosing backend
    # if CUPY_INSTALLED and isinstance(ham, HamKGPU):
    #     import cupy as xp
    # else:
    #     import numpy as xp
    import numpy as xp

    # Intra k-group Communicator for band parallelization
    kgrp_intracomm: KgrpIntraComm = wfn.pwcomm.kgrp_intracomm

    # Initializing Hamiltonian
    ham.idxspin = idxspin
    ham.ham_diag = ham.ke_gk + ham.vkb_diag + vbare_g0  # Setting preconditioner

    # Starting Here
    if wfn.bnd_par:
        numeig = wfn.numbnd_all
    else:
        numeig = wfn.numbnd
    numg = ham.numgk
    evc = xp.empty((numeig, numg), dtype=xp.complex128)
    evl = xp.empty(numeig, dtype=xp.complex128)

    numpsi = numeig
    notcnv = xp.ones(numeig, dtype=xp.bool)
    numuncnv = numeig
    evc[:] = kgrp_intracomm.psi_Allgather(wfn.evc_gk[idxspin])  # Gathering all bands
    evc[:] /= xp.linalg.norm(evc, axis=1, keepdims=True)

    # Initializing Work Arrays
    max_ndim = numwork * numeig
    _psi = xp.empty((max_ndim, numg), dtype=xp.complex128)
    _hpsi = xp.empty((max_ndim, numg), dtype=xp.complex128)
    _ham_red = xp.empty((max_ndim, max_ndim), dtype=xp.complex128)
    _ovl_red = xp.empty((max_ndim, max_ndim), dtype=xp.complex128)

    _evc_red = xp.empty((numeig, max_ndim), dtype=xp.complex128)
    evl_red = xp.empty(numeig, dtype=xp.complex128)
    scratch = xp.empty((numeig, ham.numvkb), dtype=xp.complex128)

    # Update array views when new basis vectors are added
    psi, hpsi = _psi[:numpsi], _hpsi[:numpsi]
    ham_red, ovl_red = _ham_red[:numpsi, :numpsi], _ovl_red[:numpsi, :numpsi]
    evc_red = _evc_red[:, :numpsi]

    def update_views(numpsi_: int):
        nonlocal psi, hpsi
        nonlocal ham_red, ovl_red
        nonlocal evc_red
        psi, hpsi = _psi[:numpsi_], _hpsi[:numpsi_]
        ham_red, ovl_red = _ham_red[:numpsi_, :numpsi_], _ovl_red[:numpsi_, :numpsi_]
        evc_red = _evc_red[:, :numpsi_]

    # Band-parallelization of h_psi()
    def compute_hpsi(istart: int, istop: int):
        sl = kgrp_intracomm.psi_scatter_slice(istart, istop)
        if sl.stop - sl.start > 0:
            ham.h_psi_(psi[sl], hpsi[sl], scratch)
        kgrp_intracomm.barrier()
        kgrp_intracomm.psi_Allgather_inplace(hpsi[istart:istop])
        return hpsi

    def rearrange_evc_red():
        xp.put_along_axis(evc_red, np.arange(numuncnv, dtype='i8').reshape(-1, 1),
                          evc_red[notcnv], axis=0)

        evl_red[:numuncnv] = evl_red[:numeig][notcnv]
        return evl_red, evc_red

    # Band-parallelization of generating new wavefunctions by approx inverse iteration
    def gen_new_psi():
        nonlocal numpsi
        nonlocal psi, hpsi

        sl_newpsi = slice(numpsi, numpsi + numuncnv)
        sl_bnd = kgrp_intracomm.psi_scatter_slice(0, numpsi)
        if sl_bnd.stop - sl_bnd.start > 0:
            _psi[sl_newpsi] = evc_red[:numuncnv][(slice(None), sl_bnd)] @ psi[sl_bnd]
            _hpsi[sl_newpsi] = evc_red[:numuncnv][(slice(None), sl_bnd)] @ hpsi[sl_bnd]
            kgrp_intracomm.Allreduce_sum(_psi[sl_newpsi])
            kgrp_intracomm.Allreduce_sum(_hpsi[sl_newpsi])

            _psi[sl_newpsi] *= -evl_red[:numuncnv].reshape(-1, 1)
            _psi[sl_newpsi] += _hpsi[sl_newpsi]
            ham.g_psi(_psi[sl_newpsi], evl_red[:numuncnv], in_place=True)
            _psi[sl_newpsi] /= xp.linalg.norm(_psi[sl_newpsi], axis=1, keepdims=True)

        kgrp_intracomm.barrier()
        numpsi += numuncnv
        psi = _psi[:numpsi]
        return numpsi, psi

    psi[:] = evc

    # Band-parallelization of reduced matrices calculation
    # NOTE: As only the Lower triangular matrix is used, no need to fill values in upper triangle
    def update_red(istart: int, istop: int):
        # FIXME:
        # sl = kgrp_intracomm.scatter_psi_idx(istart, istop)
        # ham_red[sl] = psi[sl].conj() @ hpsi.T
        # ovl_red[sl] = psi[sl].conj() @ psi.T
        # kgrp_intracomm.allgather_psi_inplace(ham_red)
        # kgrp_intracomm.allgather_psi_inplace(ovl_red)
        ham_red[:] = psi.conj() @ hpsi.T
        ovl_red[:] = psi.conj() @ psi.T
        return ham_red, ovl_red

    # Hermitianizing Reduced matrices
    # NOTE: As only the Lower triangular matrix is used, off-diagonal elements are not treated
    def hermitianize():
        ham_red.ravel()[:: numpsi + 1] = ham_red.ravel()[:: numpsi + 1].real
        ovl_red.ravel()[:: numpsi + 1] = ovl_red.ravel()[:: numpsi + 1].real
        return ham_red, ovl_red

    # Computing the smallest 'n_eig' eigenpairs of ham_red @ evc[:,i] = evl[i] * ovl_red @ evc[:,i]
    def compute_eig_red():
        if kgrp_intracomm.is_root:
            A, B = ham_red, ovl_red
            # _evl_red, _evc_red = eigsh(A=A, k=n_eig, M=B, which='SA', return_eigenvectors=True)
            L = xp.linalg.cholesky(B)

            L_inv = xp.linalg.inv(L)
            A_mod = L_inv @ A @ L_inv.conj().T

            _evl, _evc = xp.linalg.eigh(A_mod, "L")
            _evl, _evc = _evl[:numeig], _evc[:, :numeig]
            _evc = xp.linalg.solve(L.conj().T, _evc)

            evl_red[:numeig] = _evl
            evc_red[:] = _evc.T

        kgrp_intracomm.Bcast(_evc_red)
        kgrp_intracomm.Bcast(evl_red)
        return evl_red, evc_red

    hpsi = compute_hpsi(0, numpsi)
    ham_red, ovl_red = update_red(0, numpsi)
    ham_red, ovl_red = hermitianize()
    evl_red, evc_red = compute_eig_red()
    evl[:] = evl_red[:numeig]

    global MAXITER
    i_iter = 0
    while i_iter < MAXITER:
        evl_red, evc_red = rearrange_evc_red()
        numpsi, psi = gen_new_psi()
        update_views(numpsi)
        hpsi = compute_hpsi(numpsi - numuncnv, numpsi)
        ham_red, ovl_red = update_red(numpsi - numuncnv, numpsi)
        ham_red, ovl_red = hermitianize()
        evl_red, evc_red = compute_eig_red()
        notcnv[:] = xp.abs(evl_red - evl) > diago_thr
        numuncnv = xp.sum(notcnv)
        evl[:] = evl_red[:numeig]

        # Check if converged or need to restart
        if numuncnv == 0 or numpsi + numuncnv > max_ndim:
            evc[:] = evc_red[:numeig] @ psi
            if numuncnv == 0:
                break

            # Restart
            psi[:numeig] = evc[:]
            hpsi[:numeig] = evc_red[:numeig] @ hpsi

            numpsi = numeig
            update_views(numpsi)
            ham_red[:], ovl_red[:] = 0, 0

            ham_red[(range(numpsi), range(numpsi))] = evl
            ovl_red[(range(numpsi), range(numpsi))] = 1
            evc_red[:] = xp.eye(numpsi, dtype=xp.complex128)

        i_iter += 1

    sl = kgrp_intracomm.psi_scatter_slice(0, numeig)
    wfn.evl[idxspin] = evl[sl].real
    wfn.evc_gk[idxspin] = evc[sl]

    return {"numOuterIterations": i_iter, "numMatvecs": 0, "numPreconds": 0}
