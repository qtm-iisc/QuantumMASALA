from typing import Any, Callable
import numpy as np

from quantum_masala.core import (
    Crystal,
    GField,
    rho_check, rho_normalize,
)
from quantum_masala.pseudo import (
    loc_generate, NonlocGenerator
)
from quantum_masala.dft.pot import (
    hartree_compute, xc_compute,
    ewald_compute
)

from .wfn_bpar import WavefunBgrp, wfn_gamma_check

from quantum_masala import config


def propagate(crystal: Crystal, rho_start: GField,
              wfn_gamma: WavefunBgrp,
              xc_params: dict[str, Any],
              time_step: float, numstep: int,
              callback: Callable[[int, GField, WavefunBgrp], None],
              ):
    pwcomm = config.pwcomm
    numel = crystal.numel

    wfn_gamma_check(wfn_gamma)
    is_spin = wfn_gamma.is_spin
    is_noncolin = wfn_gamma.is_noncolin

    rho_check(rho_start, is_spin)
    grho = rho_start.gspc
    gwfn = wfn_gamma.gspc
    gkspc = wfn_gamma.gkspc

    v_ion, rho_core = GField.zeros(grho, 1), GField.zeros(grho, 1)

    l_nloc = []
    for sp in crystal.l_atoms:
        v_ion_typ, rho_core_typ = loc_generate(sp, grho)
        v_ion += v_ion_typ
        rho_core += rho_core_typ
        l_nloc.append(NonlocGenerator(sp, gwfn))
    v_ion = v_ion.to_rfield()

    en = {}

    def compute_pot_local(rho):
        v_hart, en['hart'] = hartree_compute(rho)
        v_xc, en['xc'] = xc_compute(rho, rho_core, **xc_params)
        v_loc = v_ion + v_hart + v_xc
        v_loc.Bcast()
        v_loc *= 1 / np.prod(gwfn.grid_shape)
        return v_loc
    en['ewald'] = ewald_compute(crystal, grho)

    prop_kwargs = {}
    if config.tddft_exp_method == 'taylor':
        from .expoper.taylor import TaylorExp as PropOper
        prop_kwargs['order'] = config.taylor_order
    elif config.tddft_exp_method == 'splitoper':
        from .expoper.splitoper import SplitOper as PropOper
    else:
        raise ValueError("'config.tddft_exp_method' not recognized. "
                         f"got {config.tddft_exp_method}.")

    if config.tddft_prop_method == 'etrs':
        from .prop.etrs import prop_step
    elif config.tddft_prop_method == 'splitoper':
        if config.tddft_exp_method != 'splitoper':
            raise ValueError("'config.tddft_exp_method' must be 'splitoper' when "
                             "'config.tddft_prop_method' is set to 'splitoper. "
                             f"got {config.tddft_exp_method} instead.")
        from .prop.splitoper import prop_step
    else:
        raise ValueError("'config.tddft_prop_method' not recognized. "
                         f"got {config.tddft_prop_method}.")

    v_loc = compute_pot_local(rho_start)
    prop_gamma = PropOper(gkspc, is_spin, is_noncolin, v_loc, l_nloc,
                          time_step, **prop_kwargs)

    rho = rho_start.copy()
    for istep in range(numstep):
        prop_step(wfn_gamma, rho, compute_pot_local, prop_gamma)
        rho = wfn_gamma.get_rho()
        rho_normalize(rho, numel)
        callback(istep, rho, wfn_gamma)
