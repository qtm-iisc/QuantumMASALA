__all__=["local_stress"]
import numpy as np


from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.pseudo.loc import loc_generate_dpot
from qtm.constants import RYDBERG, RY_KBAR

from qtm.dft import DFTCommMod


def stress_local(dftcomm:DFTCommMod,
                cryst:Crystal,
                 gspc:GSpace,
                 rho:FieldGType,
                 vloc:FieldGType,
                 gamma_only:bool=False):
    ## This routine calculates te stress caused by the local pseudopotential
    #setting up the characteristics of the crystal
    l_atoms = cryst.l_atoms
    numatoms=[sp.numatoms for sp in l_atoms]
    tot_num = np.sum(numatoms)
    num_typ = len(l_atoms)
    labels=np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)

    # setting up G space characteristics
    idxsort = gspc.idxsort
    numg = gspc.size_g
    cart_g = (gspc.g_cart[:,idxsort])
    if cart_g.ndim==3:
        cart_g=cart_g.reshape(cart_g.shape[0],cart_g.shape[-1])
    gtau = coords_cart_all.T @ cart_g

    ###Constructing the local pseudopotential
    v_loc = np.zeros((tot_num, numg))
    dv_loc = np.zeros((tot_num, numg))
    for isp in range(num_typ):
        v_loc[labels==isp]=np.real(vloc[isp].data)
        dv_loc_isp = loc_generate_dpot(l_atoms[isp], gspc)
        dv_loc[labels==isp]=np.real(dv_loc_isp.data)
    if dftcomm.pwgrp_intra is None or dftcomm.pwgrp_intra.size==1: v_loc=v_loc[:,idxsort]
    fact=2 if gamma_only else 1
    g_tensor=np.einsum("ij, ik->ijk", cart_g.T, cart_g.T)
    spart=2*dv_loc*np.real(np.exp(-1j*gtau)*np.conjugate(rho))
    loc_stress_dv=np.einsum("ijk, il -> iljk", g_tensor, spart.T)
    loc_stress_dv=np.sum(loc_stress_dv, axis=(0,1))
    loc_stress_v=np.eye(3)*np.sum(v_loc*np.real(np.exp(-1j*gtau)*np.conjugate(rho)))/RYDBERG
    loc_stress= loc_stress_v + loc_stress_dv
    if dftcomm.pwgrp_intra is not None:
        loc_stress = dftcomm.pwgrp_intra.allreduce(loc_stress)
    loc_stress*=fact
    return loc_stress*RY_KBAR