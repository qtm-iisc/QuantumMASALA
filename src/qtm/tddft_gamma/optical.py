import os
from typing import Optional
import numpy as np
from qtm.containers.field import FieldGType, FieldRType
from qtm.containers.wavefun import WavefunGType
from qtm.crystal.crystal import Crystal
from qtm.dft.kswfn import KSWfn
from qtm.logger import qtmlogger
from qtm.mpi.comm import QTMComm
from qtm.tddft_gamma.propagate import propagate

from scipy.integrate import simpson

DEBUGGING = True

EPS8 = 1e-8
DAMP_DEFAULT = 1e-4


def dipole_response(
    comm_world: QTMComm,
    crystal: Crystal,
    wfn_gamma: list[list[KSWfn]],
    time_step: float,
    numstep: int,
    kick_strength: float = 1e-4,
    kick_direction: str = "z",
    libxc_func: Optional[tuple[str, str]] = None,
    write_freq: int = 10,
):
    """Compute the dipole response of a system to a time-dependent kick.

    Args:
        comm_world (QTMComm): MPI communicator.
        crystal (Crystal): Crystal object.
        wfn_gamma (list[list[KSWfn]]): List of KSWfn objects.
        time_step (float): Time step for the simulation.
        numstep (int): Number of time steps.
        kick_strength (float, optional): Strength of the kick. Defaults to 1e-4.
        kick_direction (str, optional): Direction of the kick. Defaults to "z".
        libxc_func (Optional[tuple[str, str]], optional): Tuple of libxc functional and exchange-correlation functional. Defaults to None.
        write_freq (int, optional): Frequency at which to write the dipole to disk (in dipz_temp.npy file). Defaults to 10. 0 means no writing.

    Returns:
        np.ndarray: Time-dependent dipole response, i.e. dipole moment, divided by the kick strength. Units: Hartree.
    """

    rho_start = (
        wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
    )

    # Validate the inputs
    if kick_direction not in ["x", "y", "z"]:
        raise ValueError(
            "'kick_direction' must be one of 'x', 'y' or 'z'. "
            f" got '{kick_direction}'."
        )
    if kick_strength < EPS8:
        qtmlogger.warn(
            "'kick_strength' might be too small and/or negative. "
            f"got {kick_strength}."
        )

    # Create array to store the dipole at each time step
    dip_t = np.zeros((numstep + 1, 3), dtype="c16")

    # Alias the useful variables
    gspc_rho = rho_start.gspc
    gspc_wfn = wfn_gamma[0][0].gkspc.gwfn
    gkwfn = wfn_gamma[0][0].gkspc
    reallat = crystal.reallat

    # Compute the kick field
    # -----------------------
    # Compute the charge center of the ions, in cryst coordinates
    rcenter_cryst = np.zeros(3, dtype="f8", like=reallat.latvec)
    for typ in crystal.l_atoms:
        rcenter_cryst += typ.valence * np.sum(typ.r_cryst, axis=1)
    rcenter_cryst /= crystal.numel
    # Convert the charge center to cartesian coordinates
    rcenter_cart = reallat.cryst2cart(rcenter_cryst)

    # Compute the kick field mesh
    # 1. Get the mesh coordinates in cartesian coordinates (in Bohr)
    rmesh_cart = reallat.get_mesh_coords(
        *gspc_wfn.grid_shape, "cart", tuple(rcenter_cart)
    )

    def compute_dipole(
        istep: int, rho: FieldGType, write_freq: int = 10, fname: str = "dipz_temp.npy"
    ):
        r"""Compute the dipole at time step 'istep' and store it in 'dip_t'.
        $ dip = \int r \rho(r) dr = \Omega / N_{cell} \sum r_i \rho(r_i) $
        """
        # TODO: We are not getting type info and docstrings for numpy operations on Field objects.
        rho_r: FieldRType = (rho - rho_start).to_r()  # / np.prod(rho.gspc.grid_shape)

        # dip = \int r \rho(r) dr = e * \Omega / N_{cell} \sum r_i \rho(r_i)
        # r_i = (x_i, y_i, z_i) in cartesian coordinates (See defn of rmesh_cart)
        # rho is normalized to the number of electrons in the unit cell, and has units of 1 / a_0^3
        # So dip is in atomic units: a_0 * e
        # E_kick is in atomic units: Hartree / a_0 / e
        if hasattr(rho_r.gspc, "is_dist"):
            dip = rho_r.allgather().integrate_unitcell(
                np.expand_dims(rmesh_cart, axis=-4).reshape(3, 1, -1), axis=1
            )
        else:
            dip = rho_r.integrate_unitcell(
                np.expand_dims(rmesh_cart, axis=-4).reshape(3, 1, -1), axis=1
            )
        dip_t[istep + 1] = dip
        # Units: dipole is in atomic units: a_0^4 * e
        if write_freq > 0 and istep % write_freq == 0:
            qtmlogger.info(f"Step {istep}: Saving partial dipole to 'dipz.npy'.")

            if comm_world.rank == 0:
                if os.path.exists(fname) and os.path.isfile(fname):
                    os.remove(fname)
                np.save(fname, dip_t[: istep + 1] / kick_strength)

    # Store the initial dipole, i.e. the dipole before the kick
    tddft_rho_start = (
        wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
    )
    compute_dipole(-1, tddft_rho_start, write_freq=write_freq)

    # 2. Compute the phase change due to impulsive kick field
    #    (kick sttrength is in atomic units, so no conversion is needed)
    efield_kick = np.exp(
        -1j  # negative sign is for charge of electron
        * kick_strength
        * rmesh_cart[["x", "y", "z"].index(kick_direction)]
    )

    # 3. Compute the transformed wavefunction, after the kick
    # evc_r = wfn_gamma[0][0].evc_gk.to_r()
    # evc_r *= efield_kick.reshape(-1)
    # gkwfn.r2g(evc_r._data, wfn_gamma[0][0].evc_gk._data)

    if hasattr(wfn_gamma[0][0].evc_gk.gspc, "is_dist"):
        evc_r = wfn_gamma[0][0].evc_gk.to_r().allgather()
        evc_r *= efield_kick.reshape(-1)
        gkwfn.r2g(gkwfn.scatter_r(evc_r._data), wfn_gamma[0][0].evc_gk._data)
    else:
        evc_r = wfn_gamma[0][0].evc_gk.to_r()
        evc_r *= efield_kick.reshape(-1)
        gkwfn.r2g(evc_r._data, wfn_gamma[0][0].evc_gk._data)

    tddft_rho_start = (
        wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
    )

    propagate(
        comm_world,
        crystal,
        tddft_rho_start,
        wfn_gamma,
        time_step,
        numstep,
        compute_dipole,
        libxc_func,
    )

    if kick_strength == 0:
        return dip_t
    return dip_t / kick_strength


def dipole_spectrum(
    dip_t: np.ndarray,
    time_step: float,
    en_start: float = 0,
    en_end: float = 20,
    en_step: float = None,
    damp_func: str = None,
    damp_fac: float = None,
):
    """Compute the dipole spectrum from the time-dependent dipole moment data.
    All quantities are in Hartree atomic units.

    Args:
        dip_t (np.ndarray): Time-dependent dipole moment / E_{kick} data. Shape: (numstep + 1, 3)
        time_step (float): Time step used in the simulation.
        en_start (float, optional): Start of the energy range. Defaults to 0.
        en_end (float, optional): End of the energy range. Defaults to 20.
        en_step (float, optional): Energy step. Defaults to None.
        damp_func (str, optional): Damping function. Defaults to None.
        damp_fac (float, optional): Damping factor. Defaults to None.

    Returns:
        l_en (np.ndarray): List of energies (in Hartree).
        dip_en (np.ndarray): Dipole spectrum.
    """

    numstep = dip_t.shape[0] - 1
    prop_time = numstep * time_step
    if en_step is None:
        en_step = 2 * np.pi / prop_time  # in Hartree units
    time = np.linspace(0, prop_time, numstep + 1)
    l_en = np.arange(en_start, en_end, en_step)
    numen = len(l_en)

    if damp_fac is None:
        damp_func = "poly"
    if damp_func not in ["poly", "exp", "gauss"]:
        raise ValueError(
            "'damp_func' must be either 'poly', 'exp' or 'gauss'. " f"got {damp_func}."
        )
    if damp_func in ["exp", "gauss"]:
        if damp_fac is None:
            if damp_func == "exp":
                damp_fac = -np.log(DAMP_DEFAULT) / prop_time
            elif damp_func == "gauss":
                damp_fac = np.sqrt(-np.log(DAMP_DEFAULT)) / prop_time
        elif damp_fac < 0:
            raise ValueError(f"'damp_fac' must be non-negative. Got {damp_fac}")

    weight = np.empty(numstep, dtype="f8")
    if damp_func == "poly":
        x = time / prop_time
        weight = 1 - 3 * x**2 + 2 * x**3
    elif damp_func == "exp":
        weight = np.exp(-(damp_fac * time))
    elif damp_func == "gauss":
        weight = np.exp(-((damp_fac * time) ** 2))

    dip_t = dip_t * weight.reshape((-1, 1))

    dip_en = np.empty((numen, dip_t.shape[1]), dtype="c16")

    for i, en in enumerate(l_en):
        dip_en[i] = simpson(
            y=dip_t * -1j * np.sin(en * time).reshape(-1, 1), x=time, axis=0
        )

    return l_en, dip_en
