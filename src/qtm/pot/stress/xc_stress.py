import numpy as np

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.mpi.gspace import DistGSpace
from qtm.containers.field import FieldGType
from qtm.constants import RY_KBAR, RYDBERG
from qtm.config import NDArray
from qtm.dft import DFTCommMod

##LIBXC functional is set to None
def stress_xc(dftcomm:DFTCommMod,
              cryst:Crystal,
              gspc:GSpace| DistGSpace,
              rho:FieldGType,
              xc_compute:tuple
              )->NDArray:
    ## This routine calculates the stress caused by the exchange-correlation potential
    #setting up the characteristics of the crystal
    omega=cryst.reallat.cellvol
    v_xc, en_xc, GGA_stress = xc_compute
    rho_r=rho.to_r()
    size_r=gspc.size_r
    if dftcomm.pwgrp_intra is not None:
        GGA_stress = dftcomm.pwgrp_intra.allreduce(GGA_stress)
        rho_r=gspc.allgather_r(rho_r._data)
        v_xc=gspc.allgather_r(v_xc._data)
        size_r=gspc.gspc_glob.size_r
    diag_xc_r=np.real(np.sum(v_xc*np.conj(rho_r))*gspc.reallat_dv)
    diag_xc=(diag_xc_r-en_xc)
    stress_xc=np.eye(3)*diag_xc
    stress_xc/=omega
    stress_xc+=np.real(GGA_stress)/(RYDBERG*size_r)
    stress_xc/=RYDBERG
    return stress_xc*RY_KBAR



