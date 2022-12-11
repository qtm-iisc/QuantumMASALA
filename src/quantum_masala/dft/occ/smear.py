__all__ = ['compute_occ']

import numpy as np
from scipy.special import erf, erfc, xlogy

from quantum_masala.core import Wavefun
from quantum_masala import config
from quantum_masala.constants import SQRT_PI, TPI

SMEAR_THRESHOLD = 50
SMEAR_TOL = 1E-10


def gauss_occ(f: np.ndarray):
    return 0.5 * erfc(f)


def gauss_en(f: np.ndarray):
    return -0.5 * np.exp(-(f ** 2)) / SQRT_PI


def fd_occ(f: np.ndarray):
    return 1 / (1 + np.exp(f))


def fd_en(f: np.ndarray):
    occ = fd_occ(f)
    return xlogy(occ, occ) + xlogy(1 - occ, 1 - occ)


def mv_occ(f: np.ndarray):
    g = -(f + 1 / np.sqrt(2))
    return 0.5 * (erf(g) + 1) + (1 / np.sqrt(2 * np.pi)) * np.exp(-(g**2))


def mv_en(f: np.ndarray):
    g = -(f + 1 / np.sqrt(2))
    return 1 / np.sqrt(TPI) * g * np.exp(-(g**2))


def _compute_occ(l_evl: np.ndarray, e_fermi: float,
                 smear_typ: str, degauss: float):
    if smear_typ == 'gauss':
        smear_func = gauss_occ
    elif smear_typ == 'fd':
        smear_func = fd_occ
    elif smear_typ == 'mv':
        smear_func = mv_occ
    else:
        raise ValueError("'smear_typ' not recognised: "
                         f"Must be one of 'gauss', 'fd', 'mv'. Got {smear_typ}")
    f = (l_evl - e_fermi) / degauss

    occ = np.zeros_like(f)
    occ[f < -SMEAR_THRESHOLD] = 1
    mask = np.abs(f) <= SMEAR_THRESHOLD
    occ[mask] = smear_func(f[mask])
    return occ


def _compute_en(wfn: Wavefun, e_fermi: float,
                smear_typ: str, degauss: float):
    if smear_typ == 'gauss':
        smear_func = gauss_en
    elif smear_typ == 'fd':
        smear_func = fd_en
    elif smear_typ == 'mv':
        smear_func = mv_en
    else:
        raise ValueError("'smear_typ' not recognised: "
                         f"Must be one of 'gauss', 'fd', 'mv'. Got {smear_typ}")
    f = (wfn.evl - e_fermi) / degauss
    mask = np.abs(f) <= SMEAR_THRESHOLD
    return degauss * np.sum(smear_func(f[mask]))


def compute_occ(l_wfn: list[Wavefun], numel: int,
                smear_typ: str, degauss: float):
    # Check if weights of k-points add to one
    k_weight_sum = 0
    pwcomm = config.pwcomm
    if pwcomm.kgrp_rank == 0:
        k_weight_sum = pwcomm.kgrp_intercomm.allreduce_sum(
            sum(wfn.k_weight for wfn in l_wfn)
        )
    k_weight_sum = pwcomm.world_comm.bcast(k_weight_sum)
    if abs(k_weight_sum - 1) >= 1E-7:
        raise ValueError("sum of 'wfn.k_weight' across all Wavefun instances 'wfn' "
                         f"must equal 1. Got {k_weight_sum}")

    # Setting bounds for bisection method
    mu_min = min(np.amin(wfn.evl) for wfn in l_wfn)
    mu_max = max(np.amax(wfn.evl) for wfn in l_wfn)
    mu_min = pwcomm.world_comm.allreduce_min(mu_min)
    mu_max = pwcomm.world_comm.allreduce_max(mu_max)

    # Aggregating all values to prevent repeated iteration across objects
    l_evl = np.stack([wfn.evl for wfn in l_wfn])
    l_weights = np.array([wfn.k_weight for wfn in l_wfn])

    # Setting scaling factor required for calculation is un-polarised
    numspin = l_wfn[0].numspin
    fac = 2 if numspin == 1 else 1

    # Computes average occupation for given fermi level
    def compute_numel(e_mu):
        l_occ = _compute_occ(l_evl, e_mu, smear_typ, degauss)
        numel_ = fac * np.sum(l_occ * l_weights.reshape(-1, 1, 1))
        if pwcomm.kgrp_rank == 0:
            numel_ = pwcomm.kgrp_intercomm.allreduce_sum(numel_)
        return pwcomm.world_comm.bcast(numel_)

    # Bisection method implementation
    mu_guess = pwcomm.world_comm.bcast((mu_min + mu_max)/2)
    del_numel = numel - compute_numel(mu_guess)
    while abs(del_numel) > SMEAR_TOL:
        if del_numel > 0:
            mu_min = mu_guess
        else:
            mu_max = mu_guess
        mu_guess = pwcomm.world_comm.bcast((mu_min + mu_max)/2)
        del_numel = numel - compute_numel(mu_guess)

    # Broadcast and return
    e_fermi = pwcomm.world_comm.bcast(mu_guess)
    e_smear = 0.
    for wfn in l_wfn:
        wfn.occ[:] = _compute_occ(wfn.evl, e_fermi, smear_typ, degauss)
        e_smear += wfn.k_weight * _compute_en(wfn, e_fermi, smear_typ, degauss)
    if pwcomm.kgrp_rank == 0:
        e_smear = pwcomm.kgrp_intercomm.allreduce_sum(e_smear)
    e_smear = pwcomm.world_comm.bcast(e_smear)
    e_smear *= fac
    return e_fermi, e_smear
