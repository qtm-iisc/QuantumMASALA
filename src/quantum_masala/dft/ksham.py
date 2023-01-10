
from warnings import warn

import numpy as np
from scipy.linalg.blas import zgemm as gemm

from quantum_masala.core import GkSpace, RField
from quantum_masala.pseudo import NonlocGenerator


class KSHam:
    __slots__ = ["gkspc", "fft_mod", "ke_gk", "l_vkb_dij", "vnl_diag",
                 "is_noncolin", "is_spin", "vloc_r", "idxspin"]

    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: RField,
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

        self.is_noncolin = is_noncolin
        self.is_spin = is_spin
        if vloc.shape != (1 + self.is_spin, ):
            raise ValueError("'vloc.shape' must be ('numspin', ). "
                             f"got is_spin={is_spin}, vloc.shape={vloc.shape}")

        self._check_vloc(vloc)
        self.vloc_r = vloc.r
        self.idxspin = None

    def _check_vloc(self, vloc: RField):
        if not isinstance(vloc, RField):
            raise TypeError("'vloc' must be an instance of 'RField'. "
                            f"got {type(vloc)}")
        if vloc.gspc != self.gkspc.gspc:
            raise ValueError("GSpace of 'vloc' does not match that of 'self.gkspc'")
        if vloc.shape != (1 + self.is_spin, ):
            raise ValueError(f"'vloc.shape' must be ({1 + self.is_spin}). "
                             f"got {vloc.shape}")

    def set_idxspin(self, idxspin: int):
        if self.is_noncolin:
            warn("for non-collinear calculation, 'idxspin' has no effect.")
        elif not self.is_spin:
            warn("for spin-unpolarized calculation, 'idxspin' has no effect.")
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
