import numpy as np
import gc

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.constants import RYDBERG,PI
from qtm.dft import DFTCommMod

from time import perf_counter

##RYDBERG=1/2
###### The energy scale is in RYDbergs all the potential are half than Quantum Espresso.
def force_local(dftcomm: DFTCommMod,
                cryst: Crystal, 
                gspc: GSpace, 
                rho:FieldGType,
                vloc:list,
                gamma_only:bool=False):

    #Setting up characteristics of the crystal
    #start_time=perf_counter()
    l_atoms=cryst.l_atoms
    numatoms=[sp.numatoms for sp in l_atoms]
    tot_num = np.sum(numatoms)
    num_typ=len(l_atoms)
    labels=np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)

    #setting up G space characteristics
    idxsort = gspc.idxsort
    numg = gspc.size_g
    cart_g = gspc.g_cart[:,idxsort]
    if cart_g.ndim==3:
        cart_g=cart_g.reshape(cart_g.shape[0],cart_g.shape[-1])
    gtau = coords_cart_all.T @ cart_g
    omega=gspc.reallat_cellvol
    v_loc=np.zeros((tot_num, numg))
    for isp in range(num_typ):
        v_loc[labels==isp]=np.real(vloc[isp].data)
    v_loc=v_loc[:,idxsort]
    fact=2 if gamma_only else 1
    l_force=np.zeros((tot_num, 3))
    vrho=np.multiply(v_loc,(np.imag(np.exp(1j*gtau)*rho)/RYDBERG))
    l_force=np.real(vrho@cart_g.T*omega*fact)
    if l_force.ndim==3:
        l_force=l_force[0]
    if dftcomm.pwgrp_intra!=None: 
        l_force=dftcomm.pwgrp_intra.allreduce(l_force)
    #force_local=cryst.symm.symmetrize_vec(l_force)
    #force_local-=np.mean(force_local, axis=0)

    del v_loc, vrho, rho, labels, coords_cart_all, cryst, gspc, vloc, dftcomm
    gc.collect()
    return l_force