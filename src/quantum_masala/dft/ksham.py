from warnings import warn

import numpy as np
from scipy.linalg import block_diag

from quantum_masala.core import GkSpace, GField, Wavefun
from quantum_masala.pseudo import NonlocGenerator


class KSHam:
    __slots__ = ["gkspc", "ke_gk", "l_vkb", "l_vkb_H", "dij",
                 "noncolin", "numspin", "vloc_r", "idxspin", "numhpsi"]

    def __init__(self, gkspc: GkSpace, numspin: int, noncolin: bool,
                 vloc: GField,
                 l_nloc: list[NonlocGenerator]):
        self.gkspc = gkspc
        self.ke_gk = 0.5 * self.gkspc.norm2

        l_vkb_dij = [nloc.gen_vkb_dij(self.gkspc) for nloc in l_nloc]
        self.l_vkb = np.concatenate([vkb for vkb, _ in l_vkb_dij], axis=0)
        self.l_vkb_H = np.conj(self.l_vkb.T)
        self.dij = block_diag(*[dij for _, dij in l_vkb_dij])

        self.noncolin = noncolin
        self.numspin = numspin
        if vloc.shape != (2, ):
            raise ValueError("'vloc.shape' must be ('numspin', ). "
                             f"Got numspin={numspin}, vloc.shape={vloc.shape}")
        self.vloc_r = vloc.r
        self.idxspin = None
        self.numhpsi = 0

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
        self.numhpsi += l_psi.shape[0]
        l_hpsi[:] = self.ke_gk * l_psi
        l_hpsi += self.gkspc.fft_mod.r2g(
            self.vloc_r[self.idxspin]
            * self.gkspc.fft_mod.g2r(
                l_psi,
            ),
        )
        proj = l_psi @ self.l_vkb_H
        l_hpsi += (proj @ self.dij.T) @ self.l_vkb
