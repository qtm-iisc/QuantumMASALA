
import numpy as np

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.config import NDArray
from qtm.force import force_ewald, force_nonloc, force_local, force_scf
from qtm.dft import DFTCommMod 

## Addition of Ewald, Non-Local and Local forces

def force(dftcomm: DFTCommMod,
        numbnd:int,
        wavefun: tuple,
        crystal: Crystal,
        gspc: GSpace, 
        rho: FieldGType,
        vloc:list,
        nloc_dij_vkb:list,
        del_v_hxc: FieldGType,
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


    scf_force=force_scf(del_vhxc=del_v_hxc,
                        dftcomm=dftcomm,
                        cryst=crystal,
                        grho=gspc)
    if comm.rank==0 and verbosity:
        print("SCF forces are", scf_force)
    ##Add the forces
    ##Total forc
    
    force_total=np.array(ewald_force+local_force+nonlocal_force+scf_force)

    ##Make total force zero
    force_total=crystal.symm.symmetrize_vec(force_total)
    force_total-=np.mean(force_total,axis=0)

    if(remove_torque):
        #print("force after computing averages", force_total)
        ##Make the total torque zero
        # Initialize an empty matrix to store the cross products
        #print("the total  mass is", tot_mass)
        R_COM=np.sum(coords_cart_weighted,axis=0)/tot_mass
        #print("the center of mass is", R_COM)
        del_R = coords_cart_all - R_COM
        #print("the realtive coordinates are", del_R)
        del_R_norm2=np.sum(del_R**2, axis=1)
        #print("the squared norm of the relative coordinates are", del_R_norm2)
        FXR = np.empty((0, 3))

        # Compute the cross product row-wise and stack them
        for row1, row2 in zip(del_R, force_total):
            #print("the rows are", row1, row2)
            cross_product = np.cross(row1, row2)
            #print("the cross product is", cross_product)
            FXR = np.vstack([FXR, cross_product])
            #print("FXR is", FXR)

        # Compute the total torque
        torque = np.mean(FXR, axis=0)
        #print("the torque is", torque)
        #Compute the new force
        delF=np.empty((0,3))

        for row in del_R:
            cross_product=np.cross(torque,row)
            delF=np.vstack([delF,cross_product])
        #print("delF", delF)
        delF/=del_R_norm2[:,np.newaxis]
        #print("delF after division", delF)
        force_total-=delF
    ##Symmetrize the force
    
    force_total_norm=np.sqrt(np.sum(force_total**2))
    return force_total, force_total_norm


