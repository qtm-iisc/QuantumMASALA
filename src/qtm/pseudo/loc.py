from __future__ import annotations
__all__ = ['loc_generate_rhoatomic', 'loc_generate_pot_rhocore']
import numpy as np
from scipy.special import erf, spherical_jn

from qtm.crystal.basis_atoms import BasisAtoms
from qtm.gspace import GSpace
from qtm.containers import get_FieldG, FieldGType
from .upf import UPFv2Data

from qtm.config import NDArray
from qtm.msg_format import type_mismatch_msg
from qtm.logger import qtmlogger

EPS6 = 1E-6


def _simpson(f_r: NDArray, r_ab: NDArray):
    f_times_dr = f_r * r_ab
    if len(r_ab) % 2 == 0:  # If even # of points <-> odd # of interval
        f_times_dr = f_times_dr[:-1]  # Discard last point for now
    f_times_dr[:] *= 1. / 3
    f_times_dr[..., 1:-1:2] *= 4
    f_times_dr[..., 2:-1:2] *= 2
    integral = np.sum(f_times_dr, axis=-1)
    # Dealing with last interval when odd # of intervals
    if len(r_ab) % 2 == 0:
        integral += (-0.5 * f_times_dr[-2] + 4 * f_times_dr[-1])
        integral += 2.5 * (f_r[-1] * r_ab[-1] / 3.)
    return integral

def _sph2pw(r: NDArray, r_ab: NDArray, f_times_r2: NDArray, g: NDArray):
    numg = g.shape[0]
    f_g = np.empty((*f_times_r2.shape[:-1], numg), dtype='c16')
    numr = r.shape[0]

    r_ab = r_ab.copy()
    r_ab *= 1./3
    r_ab[1:-1:2] *= 4
    r_ab[2:-1:2] *= 2
    f_g[:] = spherical_jn(0, g * r[0]) * f_times_r2[..., 0] * r_ab[0]

    g = g.reshape(-1, 1)
    f_times_r2 = np.expand_dims(f_times_r2, axis=-2)
    for idxr in range(numr):
        f_g[:] += np.sum(spherical_jn(0, g * r[idxr])
                         * f_times_r2[..., idxr] * r_ab[idxr],
                         axis=-1)
    return f_g


def _check_args(sp: BasisAtoms, grho: GSpace):
    if sp.ppdata is None:
        raise ValueError(f"{BasisAtoms} instance 'sp' does not have "
                         f"pseudopotential data i.e 'sp.ppdata' is None.")
    if not isinstance(sp.ppdata, UPFv2Data):
        raise NotImplementedError("only 'UPFv2Data' supported")
    if not isinstance(grho, GSpace):
        raise TypeError(type_mismatch_msg('grho', grho, GSpace))


@qtmlogger.time('rho_generate_atomic')
def loc_generate_rhoatomic(sp: BasisAtoms, grho: GSpace) -> FieldGType:
    """Computes the electron density constructed by superposing atomic charges
    of given atomic species in crystal.

    Parameters
    ----------
    sp : BasisAtoms
        Repesents an atomic species (and corresponding atoms)
        present in the crystal.
    grho : GSpace
        G-Space of the potential/charge density.

    Returns
    -------
    rho_atomic : FieldG
        Atomic Charge density generated from given species.
    """
    _check_args(sp, grho)

    upfdata: UPFv2Data = sp.ppdata
    # Radial Mesh specified in Pseudopotential Data
    r = upfdata.r
    r_ab = upfdata.r_ab

    g_cryst = grho.g_cryst
    g_norm = grho.g_norm
    FieldG: type[FieldGType] = get_FieldG(grho)

    struct_fac = FieldG(np.sum(
        np.exp((-2 * np.pi * 1j) * (sp.r_cryst.T @ g_cryst)), axis=0
    ))

    rhoatom = upfdata.rhoatom

    rho = FieldG.empty(None)

    f_times_r2 = np.empty((1, len(r)), dtype='f8')
    f_times_r2[0] = rhoatom

    rho.data[:] = _sph2pw(r, r_ab, f_times_r2, g_norm[:])
    if grho.has_g0:
        rho.data[0] = _simpson(rhoatom, r_ab)

    rho *= struct_fac / grho.reallat_dv
    return rho


@qtmlogger.time('loc_generate')
def loc_generate_pot_rhocore(sp: BasisAtoms,
                             grho: GSpace) -> (FieldGType, FieldGType):
    """Computes the local part of pseudopotential and core electron density
    (for NLCC in XC calculation) generated by given atomic species in crystal.

    Parameters
    ----------
    sp : BasisAtoms
        Atomic species (and corresponding atoms) in the crystal.
    grho : GSpace
        G-Space of the potential/charge density

    Returns
    -------
    v_ion : FieldG
        Local part of the pseudopotenital
    rho_core : FieldG
        Charge density of core electrons
    """
    _check_args(sp, grho)

    upfdata: UPFv2Data = sp.ppdata
    # Radial Mesh specified in Pseudopotential Data
    r = upfdata.r
    r_ab = upfdata.r_ab

    # Setting constants and aliases
    cellvol = grho.reallat_cellvol
    _4pibv = 4 * np.pi / cellvol
    _1bv = 1 / cellvol

    g_cryst = grho.g_cryst
    g_norm2 = grho.g_norm2
    g_norm = np.sqrt(g_norm2)
    FieldG: type[FieldGType] = get_FieldG(grho)

    valence = upfdata.z_valence

    struct_fac = FieldG(np.sum(
        np.exp((-2 * np.pi * 1j) * (sp.r_cryst.T @ g_cryst)), axis=0
    ))

    vloc = upfdata.vloc
    if upfdata.core_correction:
        rho_atc = upfdata.rho_atc
    else:
        rho_atc = None

    v_ion = FieldG.empty(None)
    rho_core = FieldG.empty(None)

    f_times_r2 = np.empty((1 + upfdata.core_correction, len(r)), dtype='f8')
    f_times_r2[0] = (vloc * r + valence * erf(r)) * r
    if upfdata.core_correction:
        f_times_r2[1] = rho_atc * r**2

    f_g = _sph2pw(r, r_ab, f_times_r2, g_norm)
    with np.errstate(divide='ignore'):
        v_ion.data[:] = f_g[0] - valence * np.exp(-g_norm2 / 4) / g_norm2
    if grho.has_g0:
        v_ion.data[0] = _simpson(r * (r * vloc + valence), r_ab)

    if upfdata.core_correction:
        rho_core.data[:] = f_g[1]
        if grho.has_g0:
            rho_core.data[0] = _simpson(rho_atc * r**2, r_ab)
    else:
        rho_core.data[:] = 0

    N = np.prod(grho.grid_shape)
    v_ion *= _4pibv * N * struct_fac
    rho_core *= _4pibv * N * struct_fac

    return v_ion, rho_core
