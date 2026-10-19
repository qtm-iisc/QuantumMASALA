import numpy as np

from qtm.gspace.gspc import GSpace
from qtm.containers.field import FieldGType
from qtm.constants import  ELECTRON_RYD, PI, RY_KBAR
from qtm.dft import DFTCommMod

def stress_har(dftcomm:DFTCommMod,
                rho:FieldGType,
               gspc:GSpace,
               gamma_only:bool=False):

    ## Extracting the g space characteristics
    g_cart=gspc.g_cart
    norm2=gspc.g_norm2[gspc.idxsort]
    mask=norm2>1e-10
    norm2=norm2[mask]
    gnum=len(norm2)
    cart_g=g_cart[:,gspc.idxsort]
    cart_g=cart_g[:,mask]
    rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)
    rho=rho[mask]

    ##Constants
    _4pi=4*PI*ELECTRON_RYD**2

    g_tensor=2*np.einsum("ij, ik->ijk", cart_g.T, cart_g.T)/(norm2.reshape(-1,1,1))-[np.eye(3)]*gnum
    g_tensor*=((np.abs(rho)**2/norm2).reshape(-1,1,1))
    stress_har=np.sum(g_tensor, axis=0)

    fac=1 if gamma_only else 0.5
    stress_har*=-_4pi*fac
    if dftcomm.pwgrp_intra is not None:
        stress_har = dftcomm.pwgrp_intra.allreduce(stress_har)
    return stress_har*RY_KBAR