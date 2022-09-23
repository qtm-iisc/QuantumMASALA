# TODO: Check Documentation
from typing import Optional, Union, TypeVar

import numpy as np

from .mpicomm import PWComm
from .gspc import GSpace
from .gspc_wfn import GSpaceWfn
from .fft import FFTGSpaceWfc

RANDOMIZE_AMP = 0.05


class WfnKBase_:
    r"""Container for bloch wavefunctions of a given k-point.

    For a given k-point, a `GSpaceWfc` object is generated which represents
    the list of G-vectors such that the wave vector :math:`\vec{G}+\vec{k}` is within
    the KE cutoff :math:`E_{rho} / 4`. Empty arrays for storing `nbnd` wavefunctions, their
    energies and their occupations are created.

    Attributes
    ----------
    gspc : GSpace
        Represents the 'G' space truncated based on Kinetic Energy cutoff '4*ecutwfc`.
    k_cryst : tuple[float, float, float]
        k-point in crystal coords.
    k_weight : float
        weight associated to given k-point. Sum across all k-points must be normalized to 1.
    gwfc : GSpaceWfn
        Represents the 'G+k' space, truncated based on KE Cutoff 'ecutwfc'

    numspin : {1, 2}
        1 if calculation is non-polarized, 2 if spin-polarized (LSDA).
    noncolin : bool
        `True` if non-colinear calculation

    numbnd : int
        Number of bands assigned to the process' b-group

    evc_gk : list[np.ndarray]
        Array of complex numbers to store the (periodic part of the) bloch wavefunctions
        as fourier components of G vectors in `gwfc`
    evl : np.ndarray
        Array of corresponding eigenvalues of `evc_gk`
    occ : np.ndarray
        Array of corresponsing occupation number of `evc_gk`
    """

    pwcomm: PWComm = PWComm()
    gspc: GSpace = None
    numspin: int = None
    numbnd: int = None
    noncolin: bool = False

    bnd_par: bool = False

    def __init__(
        self,
        k_cryst: tuple[float, float, float],
        k_weight: float,
        idxk: int,
    ):
        self.k_cryst = np.empty(3, dtype='f8')
        self.k_cryst[:] = k_cryst
        self.k_weight = k_weight
        self.idxk = idxk

        self.gwfc = GSpaceWfn(self.gspc, self.k_cryst)
        self.fft_dri = FFTGSpaceWfc(self.gwfc)

        self.evc_gk = np.empty((self.numspin, self.numbnd, self.gwfc.numgk * (1 + self.noncolin)), dtype='c16')
        self.evl = np.empty((self.numspin, self.numbnd), dtype='f8')
        self.occ = np.empty((self.numspin, self.numbnd), dtype='f8')

    def sync(self):
        self.evc_gk = self.pwcomm.kgrp_intracomm.Bcast(self.evc_gk)
        self.evl = self.pwcomm.kgrp_intracomm.Bcast(self.evl)
        self.occ = self.pwcomm.kgrp_intracomm.Bcast(self.occ)

    def init_random_wfc(self, seed: Optional[int] = None):
        if seed is not None:
            if isinstance(seed, int):
                seed += self.idxk
            else:
                seed = None
        rng = np.random.default_rng(seed)
        rng.random(out=self.evc_gk.view('f8'))
        np.multiply(self.evc_gk.real, np.exp(2*np.pi*1j * self.evc_gk.imag), out=self.evc_gk)
        self.evc_gk /= 1 + self.gwfc.norm2

    def randomize_wfc(self, seed: Optional[int] = None):
        if seed is not None:
            if isinstance(seed, int):
                seed += self.idxk
            else:
                seed = None
        rng = np.random.default_rng(seed)

        shape = self.evc_gk.shape
        self.evc_gk *= 1 + RANDOMIZE_AMP * (rng.random(shape) + 1j * rng.random(shape))

    def compute_amp_r(self, idxspbnd: tuple[Union[list[int], slice],
                                            Union[list[int], slice]
                                            ] = (slice(None), slice(None))) -> np.ndarray:
        if not self.noncolin:
            evc_r = self.fft_dri.g2r(self.evc_gk[idxspbnd])
        else:
            raise NotImplementedError('non-colinear not implemented')
        l_amp_r = evc_r.conj() * evc_r
        l_amp_r /= np.sum(l_amp_r, axis=(-1, -2, -3), keepdims=True) * self.gspc.reallat_dv
        return l_amp_r

    @property
    def rho_r(self):
        sl = (slice(None), self.pwcomm.kgrp_intracomm.psi_scatter_slice(0, self.numbnd))
        l_amp_r = self.compute_amp_r(sl)
        rho_r = np.sum(l_amp_r * np.expand_dims(self.occ[sl], axis=(-1, -2, -3)), axis=1)
        self.pwcomm.kgrp_intracomm.Allreduce_sum(rho_r)
        return rho_r


class WfnKBgrp_(WfnKBase_):

    bnd_par = True

    def __init__(
            self,
            k_cryst: tuple[float, float, float],
            k_weight: float,
            idxk: int,
    ):
        self.numbnd_all = self.numbnd
        self.numbnd_proc = self.pwcomm.kgrp_intracomm.split_numbnd(self.numbnd)
        super().__init__(k_cryst, k_weight, idxk)

    def sync(self):
        raise ValueError("wavefunction data are distributed across processes in k-group")

    @property
    def rho_r(self):
        l_amp_r = self.compute_amp_r()
        rho_r = np.sum(l_amp_r * np.expand_dims(self.occ, axis=(-1, -2, -3)), axis=1)
        self.pwcomm.kgrp_intracomm.Allreduce_sum(rho_r)
        return rho_r


WfnK = TypeVar('WfnK', bound=Union[WfnKBase_, WfnKBgrp_])


def WavefunK(gspc_: GSpace, numspin: int, numbnd: int, noncolin: bool = False,
             dist_bands: bool = False):
    if numspin not in [1, 2]:
        raise ValueError(f"'numspin' must be either 1 or 2. Got {numspin}")

    if not isinstance(numbnd, int):
        raise ValueError(f"'numbnd' must be an integer. Got {type(numbnd)}")
    if numbnd <= 0:
        raise ValueError(f"'numbnd' must be a positive integer. Got {numbnd}")

    cls_vars = {'gspc': gspc_, 'numspin': numspin, 'numbnd': numbnd,
                'noncolin': noncolin}

    if not dist_bands:
        return type("WfnK", (WfnKBase_,), cls_vars)
    else:
        return type("WfnKBgrp", (WfnKBgrp_,), cls_vars)
