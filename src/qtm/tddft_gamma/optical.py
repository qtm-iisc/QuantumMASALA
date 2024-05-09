
from typing import Optional
import numpy as np
from qtm.containers.field import FieldGType
from qtm.containers.wavefun import WavefunGType
from qtm.crystal.crystal import Crystal
from qtm.dft.kswfn import KSWfn
from qtm.logger import qtmlogger
from qtm.mpi.comm import QTMComm
from qtm.tddft_gamma.propagate import propagate

from scipy.integrate import simpson


EPS8 = 1E-10
DAMP_DEFAULT = 1E-4


def dipole_response(comm_world: QTMComm, crystal: Crystal, rho_start: FieldGType,
                    wfn_gamma: list[list[KSWfn]],
                    time_step: float, numstep: int,
                    kick_strength: float = 1E-4, kick_direction: str = 'z',
                    libxc_func: Optional[tuple[str, str]] = None,
        ):
    """Compute the dipole response of a system to a time-dependent kick."""

    # Validate the inputs
    if kick_direction not in ['x', 'y', 'z']:
        raise ValueError("'kick_direction' must be one of 'x', 'y' or 'z'. "
                         f" got '{kick_direction}'.")
    if kick_strength < EPS8:
        qtmlogger.warn("'kick_strength' might be too small and/or negative. "
                       f"got {kick_strength}.")

    # Alias the useful variables
    gspc_rho = rho_start.gspc
    gspc_wfn = wfn_gamma[0][0].gkspc
    gkwfn = wfn_gamma[0][0].gkspc

    reallat = crystal.reallat

    # Compute the kick field
    # Compute the charge center of the ions, in cryst coordinates
    rcenter_cryst = np.zeros(3, dtype='f8')
    for typ in crystal.l_atoms:
        rcenter_cryst += typ.valence * np.sum(typ.r_cryst, axis=1)
    rcenter_cryst /= crystal.numel
    # Convert the charge center to cartesian coordinates
    rcenter_cart = reallat.cryst2cart(rcenter_cryst)

    # Compute the kick field mesh
    rmesh_cart = reallat.get_mesh_coords(*gspc_wfn.grid_shape,
                                         'cart', tuple(rcenter_cart))

    evc_r = wfn_gamma[0][0].evc_gk.to_r()
    # print("evc_r.shape", evc_r.shape)
    # print("rmesh_cart.shape", rmesh_cart.shape)
    # print(rmesh_cart[['x', 'y', 'z'].index(kick_direction)].shape)
    efield_kick = np.exp(-1j * kick_strength
                    * rmesh_cart[['x', 'y', 'z'].index(kick_direction)])

    evc_r *= efield_kick.reshape(-1)
    gkwfn.r2g(evc_r._data, wfn_gamma[0][0].evc_gk._data)

    dip_t = np.empty((numstep + 1, 3), dtype='c16')

    def compute_dipole(istep: int, rho: FieldGType, wfn_t: WavefunGType):
        r"""Compute the dipole at time step 'istep' and store it in 'dip_t'.
        $ dip = \int r \rho(r) dr $
        """
        rho_r = (rho - rho_start).to_r()
        dip = rho_r.integrate_unitcell(np.expand_dims(rmesh_cart, axis=-4).reshape(3,1,-1), axis=1)
        dip_t[istep + 1] = dip / kick_strength

    compute_dipole(-1, rho_start, wfn_gamma)
    propagate(comm_world, crystal, rho_start, wfn_gamma,
              time_step, numstep, compute_dipole, libxc_func)

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
