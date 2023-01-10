from __future__ import annotations
__all__ = ['WavefunBgrp']

from quantum_masala.core import GSpace, Wavefun, RField, GField
from quantum_masala.core.pwcomm import KgrpIntracomm
from quantum_masala.dft import KSWavefun
from quantum_masala import config, pw_logger


class WavefunBgrp(Wavefun):

    def __init__(self, gspc: GSpace, k_cryst: tuple[float, float, float],
                 k_weight: float, numbnd: int, is_spin: bool, is_noncolin: bool):
        self.kgrp_intracomm: KgrpIntracomm = config.pwcomm.kgrp_intracomm
        """Communicator for band-parallelization
        """

        if not isinstance(numbnd, int) or numbnd < 1:
            raise ValueError("'numbnd' must be a positive integer. "
                             f"Got '{numbnd}' (type {type(numbnd)})")
        if numbnd < self.kgrp_intracomm.size:
            raise ValueError("'numbnd' must be greater than the size of")
        sl = self.kgrp_intracomm.psi_scatter_slice(0, numbnd)
        self.l_ibnd = tuple(range(sl.start, sl.stop))

        super().__init__(gspc, k_cryst, numbnd, is_spin, is_noncolin)
        self.k_weight: float = k_weight
        """Weight of k-point; For evaluating integral across all k-points
        """

    @classmethod
    def from_kswfn(cls, kswfn: KSWavefun) -> WavefunBgrp:
        wfn_bgrp = cls(kswfn.gspc, tuple(kswfn.k_cryst), kswfn.k_weight,
                       kswfn.numbnd, kswfn.is_spin, kswfn.is_noncolin)
        wfn_bgrp.evc_gk[:] = kswfn.evc_gk[:, wfn_bgrp.l_ibnd]
        wfn_bgrp.occ[:] = kswfn.occ[:, wfn_bgrp.l_ibnd]
        return wfn_bgrp

    @pw_logger.time('wfn:gen_rho')
    def get_rho(self) -> GField:
        self.normalize()
        rho = RField.zeros(self.gspc, self.numspin)
        fac = 2 - self.is_spin
        for ispin in range(self.numspin):
            for ipsi in range(self.numbnd):
                rho.r[ispin] += fac * self.occ[ispin, ipsi] * \
                                self.get_amp2_r((ispin, ipsi))
        rho = rho.to_gfield()
        self.kgrp_intracomm.Allreduce_sum_inplace(rho.g)
        return rho
