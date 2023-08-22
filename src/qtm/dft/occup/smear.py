__all__ = ['compute_occ']

import numpy as np
from scipy.special import erf, erfc, xlogy

from qtm.dft.kswfn import KSWfn
from qtm.dft.config import DFTCommMod
from .utils import check_args

from qtm.constants import TPI, SQRT_PI
SMEAR_THRESHOLD = 50
EPS = 1E-10


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


l_smear_func = {
    'gauss': (gauss_occ, gauss_en),
    'fd'   : (fd_occ, fd_en),
    'mv'   : (mv_occ, mv_en),
}


def _compute_occ(l_evl: np.ndarray, e_fermi: float,
                 smear_typ: str, degauss: float):
    smear_func = l_smear_func[smear_typ][0]
    f = (l_evl - e_fermi) / degauss
    occ = np.zeros_like(f)
    occ[f < -SMEAR_THRESHOLD] = 1
    mask = np.abs(f) <= SMEAR_THRESHOLD
    occ[mask] = smear_func(f[mask])
    return occ


def _compute_en(wfn: KSWfn, e_fermi: float,
                smear_typ: str, degauss: float):
    smear_func = l_smear_func[smear_typ][1]
    f = (wfn.evl - e_fermi) / degauss
    mask = np.abs(f) <= SMEAR_THRESHOLD
    return degauss * np.sum(smear_func(f[mask]))


def compute_occ(dftcomm: DFTCommMod, l_wfn: list[list[KSWfn]], numel: int,
                is_spin: bool, smear_typ: str, degauss: float):
    # check_args(dftcomm, l_wfn, numel)
    with dftcomm.image_comm as comm:
        assert isinstance(is_spin, bool)
        assert is_spin == comm.bcast(is_spin)
        assert smear_typ in l_smear_func
        assert smear_typ == comm.bcast(smear_typ)
        assert isinstance(degauss, float)
        assert abs(degauss - comm.bcast(degauss)) < EPS

    with dftcomm.kroot_intra as comm:
        if comm.is_null:
            comm.skip_with_block()

        # Aggregating all values to prevent repeated iteration across objects
        l_evl = np.stack(wfn.evl for wfn_k in l_wfn for wfn in wfn_k)
        l_weights = np.array([wfn.k_weight for wfn_k in l_wfn for wfn in wfn_k],
                             like=l_evl)

        # Setting bounds for bisection method
        mu_min = comm.allreduce(np.amin(l_evl), comm.MIN)
        mu_max = comm.allreduce(np.amax(l_evl), comm.MAX)

        # If spin-unpolarized, numel needs to be divided by 2
        numel = comm.bcast(numel)
        if not comm.bcast(is_spin):
            numel = numel / 2

        # Computes average occupation for given fermi level
        def compute_numel(e_mu):
            l_occ = _compute_occ(l_evl, e_mu, smear_typ, degauss)
            numel_ = np.sum(l_occ * l_weights.reshape((-1, 1)))
            numel_ = comm.allreduce(numel_, comm.SUM)
            return comm.bcast(numel_)

        # Bisection method implementation
        mu_guess = (mu_min + mu_max) / 2
        del_numel = numel - compute_numel(mu_guess)
        while abs(del_numel) > EPS:
            if del_numel > 0:
                mu_min = mu_guess
            else:
                mu_max = mu_guess
            mu_min, mu_max = comm.bcast(mu_min), comm.bcast(mu_max)
            mu_guess = (mu_min + mu_max) / 2
            del_numel = numel - compute_numel(mu_guess)
        # Computing occ and e_smear
        e_fermi, e_smear = mu_guess, 0.
        for wfn_k in l_wfn:
            for wfn in wfn_k:
                wfn.occ[:] = _compute_occ(wfn.evl, e_fermi, smear_typ, degauss)
                e_smear += wfn.k_weight * _compute_en(wfn, e_fermi, smear_typ, degauss)
                dftcomm.kgrp_intra.Bcast(wfn.occ)
        e_smear = comm.allreduce(e_smear, comm.SUM)
        e_smear *= 2 if not is_spin else 1

    with dftcomm.kgrp_intra as comm:
        e_fermi = comm.bcast(e_fermi)
        e_smear = comm.bcast(e_smear)


    return e_fermi, e_smear
