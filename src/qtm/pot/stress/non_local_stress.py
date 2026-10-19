import numpy as np
from qtm.crystal import Crystal
from qtm.pseudo.nloc import NonlocGenerator
from qtm.constants import RYDBERG_HART, RY_KBAR
from qtm.config import NDArray
from qtm.dft import DFTCommMod
from qtm.mpi import scatter_slice

TOL=1e-5

def stress_nonloc(dftcomm:DFTCommMod,
                  numbnd:int,
                  wavefun:tuple,
                   cryst: Crystal,
                   nloc_dij_vkb:list
                   ) -> NDArray:
    assert isinstance(numbnd, int)
    with dftcomm.kgrp_intra as comm:
        if dftcomm.pwgrp_intra is None: band_slice=scatter_slice(numbnd, comm.size, comm.rank)
        else: band_slice=np.arange(numbnd)

    ##Getting the characteristics of the crystal
    omega=cryst.reallat.cellvol
    l_atoms = cryst.l_atoms
    num_typ=len(l_atoms)

    ##Initializing the stress tensor
    stress_nl=np.zeros((3,3)).astype(np.complex128)

    ##Looping over the wavefunctions
    for ityp in range(num_typ):
        sp=l_atoms[ityp]
        k_counter=0
        for wfn in wavefun:
            for k_wfn in wfn:
                ## Getting the evc and gk space characteristics form the wavefunction
                evc=k_wfn.evc_gk[band_slice]
                evc_data=evc.data.T
                gkspace=k_wfn.gkspc
                k_weight=k_wfn.k_weight
                gkcart=gkspace.gk_cart.T
                gknorm=gkspace.gk_norm

                occ_num=k_wfn.occ[band_slice] ##Getting the occupation numbers

                ## Getting the non-local beta projectors and dij matrices from the wavefun
                k_nonloc= NonlocGenerator(sp=sp,
                                          gwfn=gkspace.gwfn)
                vkb, dij=nloc_dij_vkb[k_counter][ityp]
                dj_vkb, dy_vkb = k_nonloc.gen_vkb_dij_deriv(k_wfn.gkspc)
                vkb=vkb.data
                dj_vkb=dj_vkb.data
                dy_vkbx, dy_vkby, dy_vkbz=dy_vkb
                dy_vkbx=dy_vkbx.data
                dy_vkby=dy_vkby.data
                dy_vkbz=dy_vkbz.data
                dy_vkb=np.array([dy_vkbx, dy_vkby, dy_vkbz])

                dij_sp=dij/RYDBERG_HART

                ## Calculation of the Diagonal Terms
                betaPsi=np.conj(vkb)@evc_data
                if dftcomm.pwgrp_intra is not None:
                    betaPsi=dftcomm.pwgrp_intra.allreduce(betaPsi)
                abs2_betaPsi=np.abs(betaPsi)**2*occ_num
                quant=np.sum(dij_sp@abs2_betaPsi)
                quant*=k_weight
                diag_stress=(np.eye(3)*quant).astype(np.complex128)
                stress_nl+=diag_stress

                ##The derivative of Spherical Bessel function
                betaPsi_d=dij_sp@betaPsi
                beta_dj=dj_vkb.T@betaPsi_d   #(shape is G,numbnd)
                Sigma_j_nl=2*np.real(np.conj(evc_data)*beta_dj)*occ_num
                Sigma_j_nl=np.sum(Sigma_j_nl, axis=1)
                gknorm_nonzero=np.where(gknorm>TOL)
                gknorm_inv=np.zeros_like(gknorm)
                gknorm_inv[gknorm_nonzero]=1/gknorm[gknorm_nonzero]
                gktensor=np.einsum('ij, ik->ijk', gkcart, gkcart)
                gktensor*=((Sigma_j_nl*gknorm_inv).reshape(-1, 1, 1))
                stress_dj=np.sum(gktensor, axis=0)*k_weight
                if dftcomm.pwgrp_intra is not None:
                    stress_dj=dftcomm.pwgrp_intra.allreduce(stress_dj)
                stress_nl+=stress_dj

                ##The derivative of spherical Harmnomics
                evc_occup=evc_data*occ_num
                beta_dy=np.array([betaPsi_d.T@dy_vkb[i] for i in range(3)])
                mult=np.conj(evc_occup.T)
                Sigma_y_nl=beta_dy*mult
                Sigma_y_diagonal = np.array([np.sum(Sigma_y_nl[i], axis=0) for i in range(Sigma_y_nl.shape[0])])
                stress_dy=2*np.real(Sigma_y_diagonal@gkcart)*k_weight
                stress_dy[0,1]=stress_dy[1,0]
                stress_dy[0,2]=stress_dy[2,0]
                stress_dy[1,2]=stress_dy[2,1]
                if dftcomm.pwgrp_intra is not None:
                    stress_dy=dftcomm.pwgrp_intra.allreduce(stress_dy)
                stress_nl+=stress_dy

    stress_nl/=omega
    stress_nl=np.real(stress_nl)
    stress_nl=cryst.symm.symmetrize_matrix(stress_nl)
    return stress_nl*RY_KBAR
