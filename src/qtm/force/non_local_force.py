import numpy as np
import gc

from qtm.crystal import Crystal
from qtm.constants import RYDBERG_HART
from qtm.config import NDArray
from qtm.dft import DFTCommMod
from qtm.mpi import scatter_slice

def force_nonloc(dftcomm:DFTCommMod,
                 numbnd: int,
                 wavefun: tuple,
                 crystal: Crystal,
                 nloc_dij_vkb:list
                 ) -> NDArray:
    """Calculate the nonlocal force on the atoma from the upf potential"""

    ##Starting of the parallelization over bands
    assert isinstance(numbnd, int)

    ##No G space parallelization
    if dftcomm.pwgrp_intra==None:
        with dftcomm.kgrp_intra as comm:
            band_slice=scatter_slice(numbnd, comm.size, comm.rank)
    else:
        band_slice=np.arange(numbnd)

    ##Getting the characteristics of the crystal
    l_atoms = crystal.l_atoms
    num_typ=len(l_atoms)
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])
    atom_label= np.concatenate([np.arange(sp.numatoms) for sp in l_atoms])

    ##Initializing the force array 
    force_nl=np.zeros((tot_atom, 3))
    k_counter=0
    for wfn in wavefun:
        for k_wfn in wfn:
            ## Getting the evc and gk-space characteristics form the wavefunctio
            evc=k_wfn.evc_gk[band_slice]
            gkspace=k_wfn.gkspc
            k_weight=k_wfn.k_weight
            gkcart=gkspace.gk_cart.T
            dij_vkb=nloc_dij_vkb[k_counter]

            ##Getting the occupation number
            occ_num=k_wfn.occ[band_slice]
            atom_counter=0
            ## Getting the non-local beta projectors and dij matrices from the wavefun
            ##It is being calculated in the root and then broadcasted over to other processors in same k group according to band parallelization
            for ityp in range(num_typ):
                sp=l_atoms[ityp]
                vkb_full, dij = dij_vkb[ityp]
                for inum in range(sp.numatoms):
                    atom_label_sp=atom_label[inum]
                    row_vkb=int(vkb_full.data.shape[0]/sp.numatoms)
                    vkb=vkb_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                    dij_sp=dij[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb, atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                    ##Constructing the G\beta\psi
                    gkcart_struc=gkcart.reshape(-1,1,3)
                    evc_sp=evc.data.T
                    Kc=gkcart_struc*evc_sp[:,:,None]
                    GbetaPsi=np.einsum("ij, jkl->ikl", vkb, np.conj(Kc))
                    GbetaPsi=dftcomm.image_comm.allreduce(GbetaPsi)
                    ##Constructing the \beta\psi
                    betaPsi=np.conj(vkb)@(evc_sp*occ_num.reshape(1,-1))
                    betaPsi=dij_sp@betaPsi
                    betaPsi=dftcomm.image_comm.allreduce(betaPsi)
                    ##Multiplying Together
                    V_NL_Psi=np.sum(GbetaPsi*betaPsi.reshape(*betaPsi.shape, 1), axis=(0,1))
                    trace=-2*np.imag(V_NL_Psi)
                    ##Multiply by Weight
                    trace = trace * k_weight
                    force_nl[atom_counter]+=trace 
                    atom_counter+=1
                    del vkb, dij_sp, GbetaPsi, betaPsi, trace, Kc, evc_sp, gkcart_struc, V_NL_Psi
                del vkb_full, dij
            del evc, gkspace, k_weight, dij_vkb
        k_counter+=1
    force_nl/=RYDBERG_HART
    force_nl=dftcomm.image_comm.allreduce(force_nl)
    force_nl/=dftcomm.image_comm.size
    #force_nl=crystal.symm.symmetrize_vec(force_nl)
    #force_nl-=np.mean(force_nl, axis=0)
    del nloc_dij_vkb, wavefun, crystal, dftcomm
    gc.collect()
    return force_nl