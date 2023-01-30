import warnings
from typing import Any, Callable
import numpy as np
from scipy.integrate import simpson

from quantum_masala.core import (
    RealLattice, Crystal,
    GField,
    rho_check, rho_normalize,
)

from .wfn_bpar import WavefunBgrp

from quantum_masala import pw_logger
from .propagate import propagate


EPS8 = 1E-10
DAMP_DEFAULT = 1E-4


def dipole_response(crystal: Crystal, rho_start: GField,
                    wfn_gamma: WavefunBgrp, xc_params: dict[str, Any],
                    time_step: float, numstep: int,
                    kick_strength: float = 1E-4, kick_direction: str = 'z',
        ):

    if kick_direction not in ['x', 'y', 'z']:
        raise ValueError("'kick_direction' must be one of 'x', 'y' or 'z'. "
                         f" got '{kick_direction}'.")
    if kick_strength < EPS8:
        pw_logger.warn("'kick_strength' might be too small and/or negative. "
                       f"got {kick_strength}.")

    gspc_rho = rho_start.gspc
    gspc_wfn = wfn_gamma.gspc
    gkwfn = wfn_gamma.gkspc

    reallat = crystal.reallat

    rcenter_cart = np.zeros(3, dtype='f8')
    for typ in crystal.l_atoms:
        rcenter_cart += typ.valence * np.sum(typ.cart, axis=1)
    rcenter_cart /= crystal.numel

    rmesh_cart = reallat.get_mesh_coords(*gspc_wfn.grid_shape, 'cart')
    rmesh_cart -= rcenter_cart.reshape((3, 1, 1, 1))

    evc_r = wfn_gamma.gkspc.fft_mod.g2r(wfn_gamma.evc_gk)
    evc_r *= np.exp(-1j * kick_strength
                    * rmesh_cart[['x', 'y', 'z'].index(kick_direction)])
    gkwfn.fft_mod.r2g(evc_r, wfn_gamma.evc_gk)

    dip_t = np.empty((numstep + 1, 3), dtype='c16')

    def compute_dipole(istep: int, rho: GField, wfn_t: WavefunBgrp):
        rho_r = (rho - rho_start).to_rfield()
        dip = rho_r.integrate(np.expand_dims(rmesh_cart, axis=-4), axis=1)
        dip_t[istep + 1] = dip / kick_strength

    compute_dipole(-1, rho_start, wfn_gamma)
    propagate(crystal, rho_start, wfn_gamma, xc_params,
              time_step, numstep, compute_dipole)

    return dip_t


def dipole_spectrum(dip_t: np.ndarray, time_step: float,
                    en_start: float = 0, en_end: float = 20, en_step: float = None,
                    damp_func: str = None, damp_fac: float = None,
                    ):
    numstep = dip_t.shape[0] - 1
    prop_time = numstep * time_step
    if en_step is None:
        en_step = 2 * np.pi / prop_time
    time = np.linspace(0, prop_time, numstep + 1)
    l_en = np.arange(en_start, en_end, en_step)
    numen = len(l_en)

    if damp_fac is None:
        damp_func = 'poly'
    if damp_func not in ['poly', 'exp', 'gauss']:
        raise ValueError("'damp_func' must be either 'poly', 'exp' or 'gauss'. "
                         f"got {damp_func}.")
    if damp_func in ['exp', 'gauss']:
        if damp_fac is None:
            if damp_func == 'exp':
                damp_fac = -np.log(DAMP_DEFAULT) / prop_time
            elif damp_func == 'gauss':
                damp_fac = np.sqrt(-np.log(DAMP_DEFAULT)) / prop_time
        elif damp_fac < 0:
            raise ValueError(f"'damp_fac' must be non-negative. Got {damp_fac}")

    weight = np.empty(numstep, dtype='f8')
    if damp_func == 'poly':
        x = time / prop_time
        weight = 1 - 3*x**2 + 2*x**3
    elif damp_func == 'exp':
        weight = np.exp(-(damp_fac * time))
    elif damp_func == 'gauss':
        weight = np.exp(-(damp_fac * time)**2)

    dip_t = dip_t * weight.reshape((-1, 1))

    dip_en = np.empty((numen, dip_t.shape[1]), dtype='c16')

    for i, en in enumerate(l_en):
        dip_en[i] = simpson(dip_t * np.exp(-1j * en * time).reshape(-1, 1),
                            time, axis=0)

    return l_en, dip_en
