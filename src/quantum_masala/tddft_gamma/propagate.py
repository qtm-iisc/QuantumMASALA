from typing import Optional, Callable
import numpy as np

from quantum_masala.core import (
    Crystal, GField, RField,
    rho_check, rho_normalize,
)
from quantum_masala.pseudo import (
    loc_generate_pot_rhocore, NonlocGenerator
)
from quantum_masala.dft.pot import (
    hartree_compute, ewald_compute,
    xc_compute, get_libxc_func, check_libxc_func,
)
from quantum_masala.dft import EnergyData

from .wfn_bpar import WavefunBgrp, wfn_gamma_check

from quantum_masala import config


def propagate(crystal: Crystal, rho_start: GField,
              wfn_gamma: WavefunBgrp,
              time_step: float, numstep: int,
              callback: Callable[[int, GField, WavefunBgrp], None],
              libxc_func: Optional[tuple[str, str]] = None,
              ):
    pwcomm = config.pwcomm
    numel = crystal.numel

    wfn_gamma_check(wfn_gamma)
    is_spin = wfn_gamma.is_spin
    is_noncolin = wfn_gamma.is_noncolin

    rho_check(rho_start, wfn_gamma.gspc, is_spin)
    gspc_rho = rho_start.gspc
    gspc_wfn = wfn_gamma.gspc
    gkspc = wfn_gamma.gkspc

    v_ion, rho_core = GField.zeros(gspc_rho, 1), GField.zeros(gspc_rho, 1)

    l_nloc = []
    for sp in crystal.l_atoms:
        v_ion_typ, rho_core_typ = loc_generate_pot_rhocore(sp, gspc_rho)
        v_ion += v_ion_typ
        rho_core += rho_core_typ
        l_nloc.append(NonlocGenerator(sp, gspc_wfn))
    v_ion = v_ion.to_rfield()

    rho_out: GField
    v_hart: RField
    v_xc: RField
    v_loc: RField

    en: EnergyData = EnergyData()

    if libxc_func is None:
        libxc_func = get_libxc_func(crystal)
    else:
        check_libxc_func(libxc_func)

    def compute_pot_local(rho_):
        nonlocal v_hart, v_xc, v_loc
        nonlocal rho_core
        v_hart, en.hartree = hartree_compute(rho_)
        v_xc, en.xc = xc_compute(rho_, rho_core, *libxc_func)
        v_loc = v_ion + v_hart + v_xc
        v_loc.Bcast()
        v_loc *= 1 / np.prod(gspc_wfn.grid_shape)
        return v_loc
    en.ewald = ewald_compute(crystal, gspc_rho)

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
        print(f'iter # {istep}')
        prop_step(wfn_gamma, rho, crystal.numel,
                  compute_pot_local, prop_gamma)
        rho = wfn_gamma.get_rho()
        rho_normalize(rho, numel)
        callback(istep, rho, wfn_gamma)
