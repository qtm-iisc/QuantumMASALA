from typing import TypeVar, Callable
import numpy as np

from quantum_masala.core import GField, RField, rho_normalize

from ..wfn_bpar import WavefunBgrp
from ..expoper.splitoper import SplitOper


def prop_step(wfn_gamma: WavefunBgrp, rho: GField,
              compute_pot_local: Callable[[GField], RField],
              prop_gamma: SplitOper):
    is_spin = wfn_gamma.is_spin
    is_noncolin = wfn_gamma.is_noncolin

    prop_gamma.oper_ke(wfn_gamma.evc_gk)
    prop_gamma.oper_nl(wfn_gamma.evc_gk, False)

    rho_half = wfn_gamma.get_rho()
    rho_half.symmetrize()
    v_loc = compute_pot_local(rho_half)

    prop_gamma.update_vloc(v_loc)
    prop_gamma.oper_vloc(wfn_gamma.evc_gk)

    prop_gamma.oper_nl(wfn_gamma.evc_gk, True)
    prop_gamma.oper_ke(wfn_gamma.evc_gk)
