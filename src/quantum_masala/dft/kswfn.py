__all__ = ['KSWavefun', 'wfn_generate', 'wfn_gen_rho']
from typing import Optional
import numpy as np

from quantum_masala.core import GSpace, KPoints, Wavefun, RField, GField
from quantum_masala.core.pwcomm import KgrpIntracomm
from quantum_masala import config, pw_logger


RANDOMIZE_AMP = 0.05


class KSWavefun(Wavefun):

    def __init__(self, gspc: GSpace, k_cryst: tuple[float, float, float],
                 k_weight: float, numbnd: int, is_spin: bool, is_noncolin: bool):
        self.kgrp_intracomm: KgrpIntracomm = config.pwcomm.kgrp_intracomm
        """Communicator for band-parallelization
        """

        super().__init__(gspc, k_cryst, numbnd, is_spin, is_noncolin)
        self.k_weight: float = k_weight
        """Weight of k-point; For evaluating integral across all k-points
        """

        self.evl: np.ndarray = np.empty((1 + self.is_spin, self.numbnd), dtype='f8')
        """(``(1 + self.is_spin, self.numbnd)``, ``'f8'``) List of KS Energy
        Eigenvalues
        """

    def sync(self, evc: bool = True, evl: bool = True, occ: bool = True):
        if evc:
            self.evc_gk = self.kgrp_intracomm.Bcast(self.evc_gk)
        if evl:
            self.evl = self.kgrp_intracomm.Bcast(self.evl)
        if occ:
            self.occ = self.kgrp_intracomm.Bcast(self.occ)

    def init_random(self, seed=None):
        seed = self.kgrp_intracomm.Bcast(seed)
        rng = np.random.default_rng(seed)
        rng.random(out=self.evc_gk.view('f8'))
        np.multiply(self.evc_gk.real, np.exp(2 * np.pi * 1j * self.evc_gk.imag),
                    out=self.evc_gk)
        self.evc_gk /= 1 + self.gkspc.norm2

    def randomize_wfc(self, seed=None):
        rng = np.random.default_rng(seed)
        shape = self.evc_gk.shape
        self.evc_gk *= 1 + RANDOMIZE_AMP * (rng.random(shape) + 1j * rng.random(shape))
        self.sync(evc=True, evl=False, occ=False)

    @pw_logger.time('wfn:get_rho')
    def get_rho(self) -> GField:
        self.normalize()
        rho = RField.zeros(self.gspc, self.numspin)
        sl = self.kgrp_intracomm.psi_scatter_slice(0, self.numbnd)
        fac = 2 - self.is_spin
        for ispin in range(self.numspin):
            for ipsi in range(sl.start, sl.stop):
                rho.r[ispin] += fac * self.occ[ispin, ipsi] * \
                                self.get_amp2_r((ispin, ipsi))
        self.kgrp_intracomm.Allreduce_sum_inplace(rho.r)
        rho = rho.to_gfield()
        return rho


def wfn_generate(gspc: GSpace, kpts: KPoints,
                 numbnd: int, is_spin: bool, is_noncolin: bool,
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
    l_wfn : list[KSWavefun]
    """
    if not isinstance(numbnd, int) or numbnd < 0:
        raise ValueError("'numbnd' must be a positive integer. "
                         f"Got {numbnd} (type {type(numbnd)})")
    if is_noncolin:
        raise ValueError(f"'noncolin = True' not yet implemmented")

    l_wfn = []
    if idxkpts is None:
        idxkpts = range(kpts.numk)
    for idxk in idxkpts:
        l_wfn.append(KSWavefun(gspc, *kpts[idxk], numbnd, is_spin, is_noncolin))

    return l_wfn


@pw_logger.time('wfn_gen_rho')
def wfn_gen_rho(l_wfn: list[KSWavefun]) -> GField:
    gspc = l_wfn[0].gspc
    numspin = l_wfn[0].numspin
    rho = GField.zeros(gspc, numspin)
    for wfn in l_wfn:
        rho += wfn.k_weight * wfn.get_rho()

    pwcomm = config.pwcomm
    if pwcomm.kgrp_rank == 0:
        pwcomm.kgrp_intercomm.Allreduce_sum_inplace(rho.g)
    pwcomm.world_comm.Bcast(rho.g)

    return rho
