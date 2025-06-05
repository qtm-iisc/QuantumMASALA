import numpy as np

from qtm.gspace.gspc import GSpace
from qtm.containers.field import FieldGType
from qtm.constants import  ELECTRON_RYD, PI, RY_KBAR

def stress_har(rho:FieldGType,
               gspc:GSpace,
               gamma_only:bool=False):

    ## Extracting the g space characteristics
    gnum=gspc.size_g
    cart_g=(gspc.g_cart.T[gspc.idxsort]).T
    norm2=gspc.g_norm2[gspc.idxsort]
    rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)

    ##Constants
    _4pi=4*PI*ELECTRON_RYD**2

    stress_har=np.zeros((3,3))
    for ig in range(1,gnum):
        gvec=cart_g[:, ig].reshape(3,-1)
        gmatrix=2*gvec@gvec.T/norm2[ig]-np.eye(3)
        gmatrix*=np.abs(rho[ig])**2/norm2[ig]
        stress_har+=gmatrix
    fac=1 if gamma_only else 0.5
    stress_har*=-_4pi*fac
    return stress_har*RY_KBAR