
__all__=["local_stress"]
import numpy as np


from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.pseudo.loc import loc_generate_pot_rhocore, loc_generate_dpot
from qtm.constants import RYDBERG, RY_KBAR


from qtm.dft import DFTCommMod
from qtm.mpi import QTMComm
from mpi4py.MPI import COMM_WORLD  



def stress_local(cryst:Crystal,
                 gspc:GSpace,
                 rho:FieldGType,
                 gamma_only:bool=False):
    ## This routine calculates te stress caused by the local pseudopotential
    #setting up the characteristics of the crystal
    l_atoms = cryst.l_atoms
    tot_num = np.sum([sp.numatoms for sp in l_atoms])
    num_typ = len(l_atoms)
    labels = np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)

    # setting up G space characteristics
    idxsort = gspc.idxsort
    numg = gspc.size_g
    cart_g = (gspc.g_cart[:,idxsort])
    gtau = coords_cart_all.T @ cart_g
    omega = gspc.reallat_cellvol


    ###Constructing the local pseudopotential
    v_loc = np.zeros((num_typ, numg))
    dv_loc = np.zeros((num_typ, numg))
    for isp in range(num_typ):
        v_loc_isp, rho_core_isp = loc_generate_pot_rhocore(l_atoms[isp], gspc, mult_struct_fact=False)
        dv_loc_isp = loc_generate_dpot(l_atoms[isp], gspc)
        v_loc[isp] = np.real(v_loc_isp.data)
        dv_loc[isp]=np.real(dv_loc_isp.data)
    v_loc = (v_loc[:,idxsort])


    loc_stress=np.zeros((3,3))
    fact=2 if gamma_only else 1

    for inum in range(tot_num):
        label=labels[inum]
        g_tensor=np.einsum("ij, ik->ijk", cart_g.T, cart_g.T)
        spart=2*dv_loc[label]*np.real(np.exp(-1j*gtau[inum])*np.conjugate(rho))
        loc_stress_v=np.sum(g_tensor*spart.reshape(-1,1,1), axis=0)
        '''for ig in range(numg):
            gidx_cart = cart_g[:, ig].reshape(3, -1)
            g_matrix = (gidx_cart @ gidx_cart.T)
            spart=2*dv_loc[label, ig]*np.real(np.exp(-1j*gtau[inum, ig])*np.conjugate(rho[ig]))
            stress_g=g_matrix*spart
            loc_stress+=stress_g'''
        loc_stress_dv=np.eye(3)*np.sum(v_loc[label]*np.real(np.exp(-1j*gtau[inum])*np.conjugate(rho)))/RYDBERG
        loc_stress_v*=fact
        loc_stress_dv*=fact
        loc_stress+=loc_stress_v+loc_stress_dv
    return loc_stress*RY_KBAR