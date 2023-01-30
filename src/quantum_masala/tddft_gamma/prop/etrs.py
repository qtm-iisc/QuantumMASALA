from typing import TypeVar, Callable
import numpy as np

from quantum_masala.core import GField, RField, rho_normalize

from ..wfn_bpar import WavefunBgrp
from ..expoper.base import TDExpOper


def prop_step(wfn_gamma: WavefunBgrp, rho: GField, numel: float,
              compute_pot_local: Callable[[GField], RField],
              prop_gamma: TDExpOper):
    evc_gk_0 = wfn_gamma.evc_gk.copy()

    v_loc = compute_pot_local(rho)
    prop_gamma.update_vloc(v_loc)
    prop_gamma.prop_psi(evc_gk_0, wfn_gamma.evc_gk)

    rho_pred = 0.5 * (rho + wfn_gamma.get_rho())
    v_loc = compute_pot_local(rho_pred)

    prop_gamma.update_vloc(v_loc)
    prop_gamma.prop_psi(evc_gk_0, wfn_gamma.evc_gk)


