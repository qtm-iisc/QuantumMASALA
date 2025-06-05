import numpy as np

from qtm.containers import FieldGType
from qtm.crystal import Crystal, BasisAtoms

from qtm.pseudo import loc_generate_rhoatomic

from qtm.gspace import GSpace
from qtm.dft import DFTCommMod

def force_scf(del_vhxc: FieldGType,
              dftcomm: DFTCommMod,
                cryst:Crystal,
                grho:GSpace,
                )->np.ndarray:

    l_atoms=cryst.l_atoms
    tot_num = np.sum([sp.numatoms for sp in l_atoms])
    num_typ=len(l_atoms)
    labels=np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    

    ##Getting the G Space characteristics
    idxsort = grho.idxsort
    numg = grho.size_g
    cart_g = (grho.g_cart[:,idxsort])
    omega=grho.reallat_cellvol
    alat=cryst.reallat.alat
    if cart_g.ndim==3:
        cart_g=cart_g.reshape(cart_g.shape[0],cart_g.shape[-1])
    struct_fact=np.exp(-1j *coords_cart_all.T @ cart_g)
    
    ###Constructing the rho corresponding to the atomic superposition.
    rho_atom=np.zeros((num_typ, numg), dtype=np.complex128)
    for isp in range(num_typ):
        rho_atom[isp]=loc_generate_rhoatomic(l_atoms[isp], grho).data
    rho_atom=rho_atom[:,idxsort]/np.prod(grho.grid_shape)

    force_scf=np.zeros((tot_num, 3))
    for atom in range(tot_num):
        label=labels[atom]
        quant=np.conj(del_vhxc) * struct_fact[atom] * rho_atom[label]

        force_scf[atom]=np.real((1j*np.sum(cart_g*quant, axis=1)))

    '''print("rho atom", rho_atom)
    print("del_vhxc", del_vhxc.data)
    print("struct_fact", struct_fact)
    print("force_scf", force_scf)'''

    if cart_g.ndim==3:
        force_scf=force_scf[0]
    
    if dftcomm.pwgrp_intra!=None:
        force_scf=dftcomm.pwgrp_intra.allreduce(force_scf)


    return force_scf
