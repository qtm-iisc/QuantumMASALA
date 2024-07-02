from copy import deepcopy
from typing import TypeVar, Callable
import numpy as np
from qtm.containers.field import FieldGType, FieldRType
from qtm.dft.kswfn import KSWfn
from qtm.tddft_gamma.expoper.base import TDExpOper



def prop_step(wfn_gamma: list[list[KSWfn]], rho: FieldGType, numel: float,
              compute_pot_local: Callable[[FieldGType], FieldRType],
              prop_gamma: TDExpOper):
    evc_gk_0 = wfn_gamma[0][0].evc_gk.copy()
    wfn_gamma_0 = wfn_gamma[0].copy()

    v_loc = compute_pot_local(rho)
    prop_gamma.update_vloc(v_loc)
    prop_gamma.prop_psi(wfn_gamma_0, wfn_gamma[0])

    rho_pred = 0.5 * (rho + wfn_gamma[0][0].compute_rho().to_g())
    v_loc = compute_pot_local(rho_pred)

    prop_gamma.update_vloc(v_loc)
    prop_gamma.prop_psi(wfn_gamma_0, wfn_gamma[0])


