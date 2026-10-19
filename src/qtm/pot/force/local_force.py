import numpy as np

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.constants import RYDBERG
from qtm.dft import DFTCommMod

###### The energy scale is in RYDbergs, all the potentials are half that of Quantum Espresso.
def force_local(dftcomm: DFTCommMod,
                cryst: Crystal,
                gspc: GSpace,
                rho:FieldGType,
                vloc:list,
                gamma_only:bool=False):

    #Setting up characteristics of the crystal
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
    vrho=np.multiply(v_loc,(np.imag(np.exp(1j*gtau)*rho)/RYDBERG))
    l_force=np.real(vrho@cart_g.T*omega*fact)
    if l_force.ndim==3:
        l_force=l_force[0]
    if dftcomm.pwgrp_intra!=None:
        l_force=dftcomm.pwgrp_intra.allreduce(l_force)
    return l_force