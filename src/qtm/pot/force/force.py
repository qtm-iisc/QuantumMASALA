from typing import Optional
import numpy as np

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.config import NDArray
from qtm.pot.force import force_ewald, force_nonloc, force_local
from qtm.dft import DFTCommMod


def force(dftcomm: DFTCommMod,
        numbnd:int,
        wavefun: tuple,
        crystal: Crystal,
        gspc: GSpace, 
        rho: FieldGType,
        vloc:list,
        nloc_dij_vkb:list,
        del_v_hxc: Optional[FieldGType | None]=None,
        gamma_only:bool=False,
        remove_torque:bool=False,
        verbosity:bool=False) -> NDArray:
    """This routine calculates the forces in Rydberg units"""
    
    l_atoms=crystal.l_atoms
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1).T
    mass_cryst=np.repeat([sp.mass for sp in l_atoms], [sp.numatoms for sp in l_atoms]).reshape(-1,1)
    coords_cart_weighted=coords_cart_all*mass_cryst
    tot_mass=np.sum(mass_cryst)

    ##Ewald force
    ewald_force=force_ewald(dftcomm=dftcomm,
                            crystal=crystal,
                           gspc=gspc,
                           gamma_only=gamma_only)

    ##Local force
    local_force=force_local(dftcomm=dftcomm,
                            cryst=crystal,
                           gspc=gspc,
                           rho=rho,
                           vloc=vloc,
                           gamma_only=gamma_only)

    ##Non-Local force
    nonlocal_force=force_nonloc(dftcomm=dftcomm,
                                numbnd=numbnd,
                                wavefun=wavefun,
                               crystal=crystal,
                               nloc_dij_vkb=nloc_dij_vkb)

    with dftcomm.image_comm as comm:
        if comm.rank==0 and verbosity:
            print("Ewald forces are", ewald_force)
            print("Local forces are", local_force)
            print("Non-Local forces are", nonlocal_force)

    ##SCF correction to the force is not yet enabled; kept at zero.
    scf_force=np.zeros_like(ewald_force)
    if comm.rank==0 and verbosity:
        print("SCF forces are", scf_force)

    ##Total force, made to sum to zero and symmetrized
    force_total=np.array(ewald_force+local_force+nonlocal_force+scf_force)
    force_total=crystal.symm.symmetrize_vec(force_total)
    force_total-=np.mean(force_total,axis=0)

    if remove_torque:
        ##Make the total torque zero
        R_COM=np.sum(coords_cart_weighted,axis=0)/tot_mass
        del_R = coords_cart_all - R_COM
        del_R_norm2=np.sum(del_R**2, axis=1)

        torque = np.mean(np.cross(del_R, force_total), axis=0)
        delF = np.cross(torque, del_R)/del_R_norm2[:,np.newaxis]
        force_total-=delF

    force_total_norm=np.sqrt(np.sum(force_total**2))
    return force_total, force_total_norm
