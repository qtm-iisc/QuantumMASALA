import numpy as np

from qtm.pot import xc
from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType, get_FieldG
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.constants import RY_KBAR, RYDBERG
from qtm.config import NDArray

##LIBXC functional is set to None
def stress_xc(cryst:Crystal,
              gspc:GSpace,
              rho:FieldGType
              )->NDArray:
    ## This routine calculates the stress caused by the exchange-correlation potential
    #setting up the characteristics of the crystal
    omega=cryst.reallat.cellvol
    l_atoms = cryst.l_atoms
    libxc_func = xc.get_libxc_func(cryst)
    #print(libxc_func)
    FieldG_rho: FieldGType = get_FieldG(gspc)
    rho_core = FieldG_rho.zeros(1)
    for sp in l_atoms:
        v_ion_sp, rho_core_sp=loc_generate_pot_rhocore(sp, gspc)
        rho_core+=rho_core_sp
    v_xc, en_xc, GGA_stress = xc.compute(rho, rho_core, *libxc_func)
    rho_r=rho.to_r()
    diag_xc_r=np.real(np.sum(v_xc._data*np.conj(rho_r._data))*gspc.reallat_dv)
    diag_xc=(diag_xc_r-en_xc)
    stress_xc=np.eye(3)*diag_xc
    stress_xc/=omega
    stress_xc+=np.real(GGA_stress)/(RYDBERG*gspc.size_r)
    stress_xc/=RYDBERG
    return stress_xc*RY_KBAR








