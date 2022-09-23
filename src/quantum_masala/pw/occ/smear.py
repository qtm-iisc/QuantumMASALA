from typing import Literal, Type
import numpy as np
from scipy.special import erf, erfc, xlogy

from quantum_masala.core import PWComm, WfnK
from quantum_masala.core.constants import PI, TPI
from .base import OccMod

SMEAR_THESHOLD = 50
SMEAR_TOL = 1e-7


def gauss_occ(f: np.ndarray):
    return 0.5 * erfc(f)


def gauss_en(f: np.ndarray):
    return -0.5 * np.exp(-(f ** 2)) / PI


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


class SmearMod(OccMod):

    def __init__(
        self,
        WfnK_: Type[WfnK], numel: float,
        smear_typ: Literal["gauss", "fd", "mv"],
        degauss: float
    ):
        super().__init__(WfnK_, numel)
        self.e_smear = 0.0

        if not isinstance(degauss, float):
            raise ValueError(f"'degauss' must be a float. Got {type(degauss)}")
        if degauss < 0:
            raise ValueError(f"'degauss' must be non-negative. Got {type(degauss)}")
        self.degauss = degauss

        if smear_typ == "gauss":
            self.smearfun_occ, self.smearfun_en = gauss_occ, gauss_en
        elif smear_typ == "fd":
            self.smearfun_occ, self.smearfun_en = fd_occ, fd_en
        elif smear_typ == "mv":
            self.smearfun_occ, self.smearfun_en = mv_occ, mv_en

    def _compute_en(self, l_evl: np.ndarray, e_mu: float, weights: np.ndarray):
        f = (l_evl - self.e_mu) / self.degauss
        mask = np.abs(f) < SMEAR_THESHOLD
        weights_ = np.broadcast_to(weights.reshape((-1, 1, 1)), mask.shape)
        e_smear_proc = np.sum(self.smearfun_en(f[mask])
                              * weights_[mask]
                              * self.degauss)
        e_smear = None
        if self.bnd_par:
            e_smear = self.pwcomm.world_comm.allreduce_sum(e_smear_proc)
        else:
            if self.pwcomm.is_kgrp_root:
                e_smear = self.pwcomm.kgrp_intercomm.allreduce_sum(e_smear_proc)
            e_smear = self.pwcomm.world_comm.bcast(e_smear)
        self.e_smear = e_smear

    def _compute_occ(self, l_evl: np.ndarray, e_mu: float, l_occ: np.ndarray):
        f = (l_evl - e_mu) / self.degauss
        mask = np.abs(f) < SMEAR_THESHOLD

        l_occ[:] = np.heaviside(-f, 0, out=l_occ, where=~mask)
        l_occ[mask] = self.smearfun_occ(f[mask])
        return l_occ

    def compute(self, l_wfnk: list[WfnK]):
        l_evl = np.stack([wfn.evl for wfn in l_wfnk])
        l_occ = np.empty_like(l_evl, dtype='f8')
        weights = np.array([wfn.k_weight for wfn in l_wfnk], dtype='f8')

        e_mu_guess, e_mu_min, e_mu_max = None, None, None
        if self.bnd_par:
            e_mu_min = self.pwcomm.world_comm.allreduce_scalar_min(np.amin(l_evl))
            e_mu_max = self.pwcomm.world_comm.allreduce_scalar_max(np.amax(l_evl))
        else:
            if self.pwcomm.is_kgrp_root:
                e_mu_min = self.pwcomm.kgrp_intercomm.allreduce_min(np.amin(l_evl))
                e_mu_max = self.pwcomm.kgrp_intercomm.allreduce_max(np.amax(l_evl))
        e_mu_min = self.pwcomm.world_comm.bcast(e_mu_min)
        e_mu_max = self.pwcomm.world_comm.bcast(e_mu_max)
        e_mu_guess = (e_mu_min + e_mu_max) / 2

        self._compute_occ(l_evl, e_mu_guess, l_occ)
        numel = self._compute_numel(l_occ, weights)

        del_numel = self.numel - numel
        while abs(del_numel) > SMEAR_TOL:
            if del_numel > 0:
                e_mu_min = e_mu_guess
            else:
                e_mu_max = e_mu_guess

            e_mu_guess = (e_mu_min + e_mu_max) / 2
            e_mu_guess = self.pwcomm.world_comm.bcast(e_mu_guess)
            self._compute_occ(l_evl, e_mu_guess, l_occ)
            numel = self._compute_numel(l_occ, weights)
            del_numel = self.numel - numel

        self.e_mu = self.pwcomm.world_comm.bcast(e_mu_guess)

        self._compute_occ(l_evl, e_mu_guess, l_occ)
        for ik, wfn in enumerate(l_wfnk):
            wfn.occ[:] = l_occ[ik]

        self._compute_en(l_evl, self.e_mu, weights)
        return l_wfnk
