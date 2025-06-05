import numpy as np

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.config import NDArray
from qtm.dft import DFTCommMod


from . import stress_ewald, stress_local, stress_kinetic, stress_har, stress_nonloc, stress_xc
from qtm.constants import RY_KBAR

#Ewald input, crystal, GSpace, gamma_only(optional)
#Hartree input rho, gspace, gamma_only(optional)
#kinetic input crystal, wfn_k_group
# local inputs, crystal, gspc, rho, gamma_only
# xc inputs crystal, gspc, rho
#non local inputs wavefun, crysal


def stress(dftcomm:DFTCommMod,
            cryst: Crystal, 
           gspc:GSpace,
           rho:FieldGType,
           numbnd:int,
           wfn_k_group:tuple,
           gamma_only:bool=False,
           verbosity:bool=False,
           unit: str = 'Ry/bohr**3'
           )->NDArray:
    
    ##summation of all the stresses

    ewald_stress=stress_ewald(crystal=cryst,
                                gspc=gspc,
                                gamma_only=gamma_only)
    
    local_stress=stress_local(cryst=cryst,
                                gspc=gspc,
                                rho=rho,
                                gamma_only=gamma_only)

    
    kinetic_stress=stress_kinetic(dftcomm=dftcomm,
                                numbnd=numbnd,
                                  cryst=cryst,
                                wfn_k_group=wfn_k_group)

    
    hartree_stress=stress_har(rho=rho,
                            gspc=gspc)

    
    stress_nl=stress_nonloc(dftcomm=dftcomm,
                            numbnd=numbnd,
                            wavefun=wfn_k_group,
                            cryst=cryst)

    
    xc_stress= stress_xc(cryst=cryst,
                            gspc=gspc,
                            rho=rho)
    
    stress_total_unsym=ewald_stress+local_stress+kinetic_stress+hartree_stress+stress_nl+xc_stress

    stress_total=cryst.symm.symmetrize_matrix(stress_total_unsym)

    with dftcomm.image_comm as comm:
        if comm.rank==0 and verbosity:
            print("stress ewald", ewald_stress)
            print("local stress", local_stress)
            print("kinetic stress", kinetic_stress)
            print("har stress", hartree_stress)
            print("non local stress", stress_nl)
            print("xc stress", xc_stress)
            print("stress before symmetrization", stress_total_unsym)
            print("stress after symmetrization", stress_total)

    if unit == 'Ry/bohr**3':
        stress_total = stress_total/RY_KBAR
    elif unit== 'kbar':
        stress_total = stress_total
    else:
        raise ValueError('unit should be either `Ry/bohr**3` or `kbar`')
    total_presure=np.trace(stress_total)/3
    return stress_total, total_presure





