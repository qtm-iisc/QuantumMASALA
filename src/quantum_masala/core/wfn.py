__all__ = ['Wavefun', 'wfn_generate', 'wfn_gen_rho']

from typing import Optional
import numpy as np

from quantum_masala.core import GSpace, GkSpace, GField, RField, KPoints
from quantum_masala.core.pwcomm import KgrpIntracomm
from quantum_masala import config, pw_counter


RANDOMIZE_AMP = 0.05


class Wavefun:
    """Container for Bloch Wavefunctions that are the eigenkets of the
    Kohn-Sham Hamiltonian in Plane-Wave Basis.

    Parameters
    ----------
    gspc : GSpace
        G-Space representing the smooth grid for wavefunctions. Note that its
        cutoff must be 4 times the wavefunction Kinetic Energy cutoff 'ecutwfc'
    k_cryst: tuple[float, float, float]
        Crystal coordinates of the k-point
    k_weight: float
        Weight of the input k-point; For Integrating across the first Brillouin
        Zone of the Reciprocal Lattice
    numbnd : int
        Number of bands to store
    noncolin : bool
        Non-collinear calculation yet to be implemented
    """
    def __init__(self, gspc: GSpace, k_cryst: tuple[float, float, float],
                 k_weight: float, numspin: int, numbnd: int, noncolin: bool):
        pw_counter.start_timer('wfn:init')
        self.kgrp_intracomm: KgrpIntracomm = config.pwcomm.kgrp_intracomm
        """PW Communicator object
        """
        self.k_cryst: np.ndarray = np.empty(3, dtype='f8')
        """(``(3, )``, ``'f8'``) Crystal Coordinates of k-point
        """
        self.k_cryst[:] = k_cryst

        self.k_weight: float = k_weight
        """Weight of k-point; For integrating quantities across all k-points
        """

        self.gspc: GSpace = gspc
        """Represents the smooth FFT grid for wavefunctions
        """
        self.gkspc: GkSpace = GkSpace(self.gspc, self.k_cryst)
        r"""Represents the set of :math:`\mathbf{G}+\mathbf{k}` vectors within
        Kinetic energy cutoff 'ecutwfc' = ``self.gspc.ecut / 4``
        """

        self.numspin: int = numspin
        """Number of spin states
        """
        self.numbnd: int = numbnd
        """Number of bands
        """
        self.noncolin: bool = noncolin
        """To be implemented
        """
        self.evc_gk: np.ndarray = np.empty((self.numspin, self.numbnd,
                                            self.gkspc.numgk * (1 + self.noncolin)),
                                           dtype='c16')
        """(``(self.numspin self.numbnd, self.gkspc.numgk)``, ``'c16'``)
        List of wavefunctions in PW Basis described by ``self.gkspc``
        """
        self.evl: np.ndarray = np.empty((self.numspin, self.numbnd), dtype='f8')
        """(``(self.numspin self.numbnd)``, ``'f8'``) List of energy eigenvalues
        """
        self.occ: np.ndarray = np.empty((self.numspin, self.numbnd), dtype='f8')
        """(``(self.numspin self.numbnd)``, ``'f8'``) List of occupation
        numbers
        """
        self.sync()
        pw_counter.stop_timer('wfn:init')

    def sync(self, evc: bool = True, evl: bool = True, occ: bool = True):
        if evc:
            self.evc_gk = self.kgrp_intracomm.Bcast(self.evc_gk)
        if evl:
            self.evl = self.kgrp_intracomm.Bcast(self.evl)
        if occ:
            self.occ = self.kgrp_intracomm.Bcast(self.occ)

    def init_random(self, seed=None):
        rng = np.random.default_rng(seed)
        rng.random(out=self.evc_gk.view('f8'))
        np.multiply(self.evc_gk.real, np.exp(2*np.pi*1j * self.evc_gk.imag),
                    out=self.evc_gk)
        self.evc_gk /= 1 + self.gkspc.norm2
        self.sync(evc=True, evl=False, occ=False)

    def randomize_wfc(self, seed=None):
        rng = np.random.default_rng(seed)
        shape = self.evc_gk.shape
        self.evc_gk *= 1 + RANDOMIZE_AMP * (rng.random(shape) + 1j * rng.random(shape))
        self.sync(evc=True, evl=False, occ=False)

    def normalize(self):
        self.evc_gk /= np.linalg.norm(self.evc_gk, axis=-1, keepdims=True) \
                       * np.sqrt(self.gspc.reallat_cellvol) \
                       / np.prod(self.gspc.grid_shape)

    def compute_amp_r(self, l_idxbnd) -> np.ndarray:
        if not self.noncolin:
            evc_r = self.gkspc.fft_mod.g2r(self.evc_gk[l_idxbnd])
        else:
            raise NotImplementedError('non-colinear not implemented')
        l_amp_r = evc_r.conj() * evc_r
        return l_amp_r

    def get_rho(self) -> GField:
        pw_counter.start_timer('wfn:gen_rho')
        self.normalize()
        rho = RField.zeros(self.gspc, self.numspin)
        sl = self.kgrp_intracomm.psi_scatter_slice(0, self.numbnd)
        fac = (2 if self.numspin == 1 else 1)
        for ispin in range(self.numspin):
            for ipsi in range(sl.start, sl.stop):
                rho.r[ispin] += fac * self.occ[ispin, ipsi] * \
                                self.compute_amp_r((ispin, ipsi))
        self.kgrp_intracomm.Allreduce_sum_inplace(rho.r)
        pw_counter.stop_timer('wfn:gen_rho')
        return rho.to_gfield()


def wfn_generate(gspc: GSpace, kpts: KPoints,
                 numspin: int, numbnd: int, noncolin: bool,
                 idxkpts: Optional[list[int]] = None):
    """Generates ``Wavefun`` instances for each k-point in ``kpts``.


    Parameters
    ----------
    gspc : GSpace
    kpts : KPoints
    numspin : int
    numbnd : int
    noncolin : bool

    Returns
    -------
    l_wfn : list[Wavefun]
    """
    pw_counter.start_timer('wfn_generate')
    if numspin not in [1, 2]:
        raise ValueError(f"'numspin' must be either 1 or 2. Got {numspin}")
    if not isinstance(numbnd, int) or numbnd < 0:
        raise ValueError(f"'numbnd' must be a positive integer. Got {numbnd}")
    if noncolin:
        raise ValueError(f"'noncolin = True' not yet implemmented")

    l_wfn = []
    if idxkpts is None:
        idxkpts = range(kpts.numk)
    for idxk in idxkpts:
        l_wfn.append(Wavefun(gspc, *kpts[idxk], numspin, numbnd, noncolin))

    pw_counter.stop_timer('wfn_generate')
    return l_wfn


def wfn_gen_rho(l_wfn: list[Wavefun]) -> RField:
    pw_counter.start_timer('wfn_gen_rho')
    gspc = l_wfn[0].gspc
    numspin = l_wfn[0].numspin
    rho = GField.zeros(gspc, numspin)
    for wfn in l_wfn:
        rho += wfn.k_weight * wfn.get_rho()

    pwcomm = config.pwcomm
    if pwcomm.kgrp_rank == 0:
        pwcomm.kgrp_intercomm.Allreduce_sum_inplace(rho.g)
    pwcomm.world_comm.Bcast(rho.g)

    pw_counter.stop_timer('wfn_gen_rho')
    return rho
