# TODO: Check Documentation
from typing import Optional

import numpy as np

from .mpicomm import PWComm
from .gspc import GSpace
from .gspc_wfc import GSpaceWfc
from .fft import FFTGSpaceWfc
from .kpts import KPointsKgrp

RANDOMIZE_AMP = 0.05


class ElectronWfc:
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
        weight associated to given k-point
    gwfc : GSpaceWfc
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

    def __init__(
        self,
        gspc: GSpace,
        k_cryst: tuple[float, float, float],
        k_weight: float,
        idxk: int,
        numspin: int,
        numbnd: int,
        noncolin: bool = False,
    ):
        self.gspc = gspc

        self.idxk = idxk
        self.k_cryst = np.empty(3, dtype='f8')
        self.k_cryst[:] = k_cryst
        self.k_weight = k_weight

        self.gwfc = GSpaceWfc(gspc, self.k_cryst)
        self.fft_dri = FFTGSpaceWfc(self.gwfc)

        if numspin not in [1, 2]:
            raise ValueError(f"'numspin' must be either 1 or 2. Got {numspin}")
        self.numspin = 1 if numspin == 1 else 2
        self.noncolin = noncolin

        if not isinstance(numbnd, int):
            raise ValueError(f"'numbnd' must be an integer. Got {type(numbnd)}")
        if numbnd <= 0:
            raise ValueError(f"'numbnd' must be a positive integer. Got {numbnd}")

        self.numbnd = numbnd

        self.evc_gk = np.empty((self.numspin, self.numbnd, self.gwfc.numgk * (1 + self.noncolin)), dtype='c16')
        self.evl = np.empty((self.numspin, self.numbnd), dtype='f8')
        self.occ = np.empty((self.numspin, self.numbnd), dtype='f8')

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

    def compute_rho_r(self) -> np.ndarray:
        evc_r = self.fft_dri.g2r(self.evc_gk)
        l_rho_r = evc_r.conj() * evc_r
        l_rho_r /= np.sum(l_rho_r, axis=(-1, -2, -3), keepdims=True) * self.gspc.reallat_dv
        rho_r = np.sum(l_rho_r * np.expand_dims(self.occ, axis=(-1, -2, -3)), axis=1)
        return rho_r


class ElectronWfcBgrp(ElectronWfc):

    pwcomm: PWComm = PWComm()

    def __init__(
            self,
            gspc: GSpace,
            k_cryst: tuple[float, float, float],
            k_weight: float,
            idxk: int,
            numspin: int,
            numbnd: int,
            noncolin: bool = False,
    ):
        self.numbnd_all = numbnd
        self.numbnd_proc = self.pwcomm.kgrp_intracomm.split_numbnd(numbnd)
        super().__init__(gspc, k_cryst, k_weight, idxk, numspin, self.numbnd_proc, noncolin)
