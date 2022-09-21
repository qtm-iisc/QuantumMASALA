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
    r"""Container for bloch wavefunctions of given bands across input k-points

    For given list of k-points, a `GSpaceWfc` object is generated which represents the
    (unique) G-Sphere for each k-point. Empty arrays for storing `nbnd` wavefunctions are
    created for each k-point. All eigenvalues are stored in a separate single array.

    Attributes
    ----------
    gspc : GSpace
        Represents the G-space truncated based on Kinetic Energy cutoff '4*ecutwfc`
    kpts : KPointsKgrp
        Input list of k-points. Part of the full set of k-points distributed across k-groups.
    gwfc : GSpaceWfc
        Represents the G-Sphere (basis) of all k-points, truncated based on KE Cutoff 'ecutwfc'

    numspin : {1, 2}
        1 if calculation is non-polarized, 2 if spin-polarized (LSDA).
    noncolin : bool
        `True` if non-colinear calculation. Not implemented

    numbnd : int
        Number of bands assigned to the process' b-group

    l_evc_gk : list[np.ndarray]
        List of numpy arrays of complex numbers to store the eigen-wavefunctions
        in reciprocal space :math:`\mathbf{G} + \mathbf{k}` for each k-point in `kgrp`
    l_evl : np.ndarray
        NumPy array containing list of all eigenvalues of `l_evc_gk`
    """

    pwcomm: PWComm = PWComm()

    def __init__(
        self,
        gspc: GSpace,
        k_cryst: list[float, float, float],
        idxk: int,
        numspin: int,
        numbnd: int,
        noncolin: bool = False,
    ):
        self.gspc = gspc

        self.idxk = idxk
        self.k_cryst = np.empty(3, dtype='f8')
        self.k_cryst[:] = k_cryst

        self.gwfc = GSpaceWfc(gspc, self.k_cryst)
        self.fft_dri = FFTGSpaceWfc(self.gwfc)

        if numspin not in [1, 2]:
            raise ValueError(f"'numspin' must be either 1 or 2. Got {numspin}")
        if noncolin:
            raise NotImplementedError(f"Non-collinear Calculation not implemented")
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


ElectronWfcBgrp = ElectronWfc