
from warnings import warn

import numpy as np
from scipy.linalg.blas import zgemm as gemm

from quantum_masala.core import GkSpace, GField, Wavefun
from quantum_masala.pseudo import NonlocGenerator


class KSHam:
    __slots__ = ["gkspc", "fft_mod", "ke_gk", "l_vkb_dij", "vnl_diag",
                 "noncolin", "numspin", "vloc_r", "idxspin"]

    def __init__(self, gkspc: GkSpace, numspin: int, noncolin: bool,
                 vloc: GField,
                 l_nloc: list[NonlocGenerator]):
        self.gkspc = gkspc
        self.fft_mod = self.gkspc.fft_mod
        self.ke_gk = 0.5 * self.gkspc.norm2

        self.l_vkb_dij = []
        self.vnl_diag = np.zeros(self.gkspc.numgk, dtype='c16')
        for nloc in l_nloc:
            vkb, dij, vkb_diag = nloc.gen_vkb_dij(self.gkspc)
            self.l_vkb_dij.append((vkb, dij))
            self.vnl_diag += vkb_diag

        self.noncolin = noncolin
        self.numspin = numspin
        if vloc.shape != (self.numspin, ):
            raise ValueError("'vloc.shape' must be ('numspin', ). "
                             f"Got numspin={numspin}, vloc.shape={vloc.shape}")
        self.vloc_r = vloc.r
        self.idxspin = None

    @classmethod
    def from_wfn(cls, wfn: Wavefun, vloc: GField, l_nloc: list[NonlocGenerator]):
        return cls(wfn.gkspc, wfn.numspin, wfn.noncolin, vloc, l_nloc)

    def set_idxspin(self, idxspin: int):
        if self.numspin == 1:
            if idxspin != 0:
                raise ValueError(f"'idxspin' must be 0 for numspin=1. Got {idxspin}")
        elif self.noncolin:
            warn("For non-collinear calculation, the value of 'idxspin' has no effect")
        elif idxspin not in [0, 1]:
            raise ValueError(f"'idxspin' must be 0 or 1 for numspin=2. Got {idxspin}")
        self.idxspin = idxspin

    def h_psi(self, l_psi: np.ndarray, l_hpsi: np.ndarray):
        l_hpsi[:] = self.ke_gk * l_psi
        for ipsi in range(l_psi.shape[0]):
            l_hpsi[ipsi] += self.fft_mod.r2g(
                self.vloc_r[self.idxspin]
                * self.fft_mod.g2r(
                    l_psi[ipsi],
                ),
            )

        for vkb, dij in self.l_vkb_dij:
            # proj = vkb.conj() @ l_psi.T
            # l_hpsi += (dij @ proj).T @ vkb
            proj = gemm(alpha=1.0, a=vkb.T, trans_a=2, b=l_psi.T, trans_b=0,
                        )
            gemm(alpha=1.0, a=vkb.T, trans_a=0, b=dij@proj, trans_b=0,
                 beta=1.0, c=l_hpsi.T, overwrite_c=True)
