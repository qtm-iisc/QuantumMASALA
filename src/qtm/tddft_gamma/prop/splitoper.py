from typing import TypeVar, Callable
import numpy as np

# from quantum_masala.core import GField, RField, rho_normalize
from qtm.containers.field import FieldGType, FieldRType
from qtm.dft.kswfn import KSWfn
from qtm.tddft_gamma.expoper.base import TDExpOper

# from ..wfn_bpar import WavefunBgrp
from ..expoper.splitoper import SplitOper

def normalize_rho(rho: FieldGType, numel: float):
    rho *= numel / (sum(rho.data_g0) * rho.gspc.reallat_dv)


def prop_step(    
    wfn_gamma: list[list[KSWfn]],
    rho: FieldGType,
    numel: float,
    compute_pot_local: Callable[[FieldGType], FieldRType],
    prop_gamma: SplitOper
    ):
    # is_spin = wfn_gamma.is_spin
    # is_noncolin = wfn_gamma.is_noncolin

    
    prop_gamma.oper_ke(wfn_gamma[0])
    prop_gamma.oper_nl(wfn_gamma[0], reverse=False)

    # rho_half = wfn_gamma.get_rho()
    # rho_half = wfn_gamma[0][0].compute_rho(ret_raw=True).to_g() 
    rho_half = wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
    normalize_rho(rho_half, numel)
    # rho_half = rho_normalize(rho_half, numel)
    v_loc = compute_pot_local(rho_half)

    prop_gamma.update_vloc(v_loc)
    prop_gamma.oper_vloc(wfn_gamma[0])

    prop_gamma.oper_nl(wfn_gamma[0], reverse=True)
    prop_gamma.oper_ke(wfn_gamma[0])

    rho = wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
