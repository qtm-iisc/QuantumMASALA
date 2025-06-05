from copy import deepcopy
from typing import TypeVar, Callable
import numpy as np
from qtm.containers.field import FieldGType, FieldRType
from qtm.dft.kswfn import KSWfn
from qtm.tddft_gamma.expoper.base import TDExpOper


def normalize_rho(rho: FieldGType, numel: float):
    return


def prop_step(
    wfn_gamma: list[list[KSWfn]],
    rho: FieldGType,
    numel: float,
    compute_pot_local: Callable[[FieldGType], FieldRType],
    prop_gamma: TDExpOper,
):
    """
    Perform an enforced time reversal symmetry (ETRS) propagation step in the time-dependent density functional theory (TDDFT) calculation.

    Args:
        wfn_gamma (list[list[KSWfn]]): List of wavefunctions at different time steps.
        rho (FieldGType): Density field.
        numel (float): Number of electrons.
        compute_pot_local (Callable[[FieldGType], FieldRType]): Function to compute the local potential.
        prop_gamma (TDExpOper): Time-dependent exponential operator.

    Returns:
        None

    Note:
        1. This function modifies the input wavefunctions in place.
        2. This method is probably closer to the exponential midpoint rule than to the enforced time reversal symmetry (ETRS) method.

    Algorithm:
        1. Store the wavefunction at current time step.
        2. Update the local potential, to be compatible with the current density.
        3. Propagate the wavefunction using the local potential at the current time step.
        4. Find the density at the half step, by interpolating between (i.e. simply averaging) the density at the current time step and the density at the next time step.
        5. Use the interpolated density to compute the local potential at the half step, and update the Hamiltonian.
    """

    # Store the wavefunction at the previous time step.
    wfn_gamma_prev = [
        KSWfn(
            wfn_gamma[0][i].gkspc,
            wfn_gamma[0][i].k_weight,
            wfn_gamma[0][i].numbnd,
            wfn_gamma[0][i].is_noncolin,
        )
        for i in range(len(wfn_gamma[0]))
    ]
    for i in range(len(wfn_gamma[0])):
        wfn_gamma_prev[i].evc_gk._data[:] = wfn_gamma[0][i].evc_gk._data.copy()
        wfn_gamma_prev[i].evl[:] = wfn_gamma[0][i].evl
        wfn_gamma_prev[i].occ = wfn_gamma[0][i].occ

    rho = wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
    normalize_rho(rho, numel)
    v_loc = compute_pot_local(rho)
    prop_gamma.update_vloc(v_loc)
    prop_gamma.prop_psi(wfn_gamma_prev, wfn_gamma[0])

    rho_pred = (
        wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
    )
    normalize_rho(rho_pred, numel)
    rho_pred._data = 0.5 * (rho._data + rho_pred._data)

    # Use the interpolated rho to compute the local potential at the half step,
    # and update the Hamiltonian.
    v_loc = compute_pot_local(rho_pred)
    prop_gamma.update_vloc(v_loc)

    # Propagate the wavefunction using the approximated Hamiltonian at half step
    prop_gamma.prop_psi(wfn_gamma_prev, wfn_gamma[0])
    rho = wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
