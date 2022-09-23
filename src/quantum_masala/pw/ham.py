import numpy as np
from scipy.linalg import block_diag

from quantum_masala.core import WfnK, FFTGSpaceWfc

from .pot.pseudo import PPDataNonLocal


class HamK:
    idxspin: int
    numgk: int
    fft_dri: FFTGSpaceWfc
    ke_gk: np.ndarray
    vloc_r: np.ndarray
    l_vkb: np.ndarray
    l_vkb_H: np.ndarray
    numvkb: int
    dij: np.ndarray
    vkb_diag: np.ndarray

    __slots__ = ["idxspin", "numgk", "fft_dri", "ke_gk", "vloc_r",
                 "l_vkb", "l_vkb_H", "numvkb", "dij", "vkb_diag",
                 "ham_diag"
                 ]

    def __init__(
        self,
        wfn: WfnK,
        vloc_r: np.ndarray,
        l_ppnl: list[PPDataNonLocal],
    ):
        self.idxspin = 0

        gwfc = wfn.gwfc
        self.numgk = gwfc.numgk
        self.fft_dri = wfn.fft_dri

        self.ke_gk = 0.5 * gwfc.norm2
        self.vloc_r = vloc_r

        l_vkb_dij = [ppnl.gen_vkb_dij(gwfc) for ppnl in l_ppnl]
        self.l_vkb = np.concatenate([vkb for vkb, _ in l_vkb_dij], axis=0)
        self.numvkb = self.l_vkb.shape[0]
        self.l_vkb_H = np.conj(self.l_vkb.T)

        self.dij = block_diag(*[dij for _, dij in l_vkb_dij])
        self.vkb_diag = np.sum(
            np.diag(self.dij).reshape(-1, 1) * (self.l_vkb.conj() * self.l_vkb), axis=0
        )

        self.ham_diag = None

    def h_psi_(self, l_psi: np.ndarray, l_hpsi: np.ndarray, scratch: np.ndarray):
        numpsi = l_psi.shape[0]
        if numpsi == 0:
            return
        np.matmul(l_psi, self.l_vkb_H, out=scratch[:numpsi])
        np.matmul(scratch[:numpsi], self.dij.T, out=scratch[:numpsi])
        np.matmul(scratch[:numpsi], self.l_vkb, out=l_hpsi)
        l_hpsi += self.ke_gk * l_psi
        l_hpsi += self.fft_dri.r2g(
            self.vloc_r[self.idxspin]
            * self.fft_dri.g2r(
                l_psi,
            ),
        )

    def h_psi(self, l_psi: np.ndarray):
        h_psi = 0.5 * self.ke_gk * l_psi
        h_psi += self.fft_dri.r2g(
            self.vloc_r[self.idxspin]
            * self.fft_dri.g2r(
                l_psi,
            ),
        )
        proj = self.l_vkb.conj() @ l_psi.T
        h_psi += (self.dij @ proj).T @ self.l_vkb

    def g_psi(self, l_psi: np.ndarray, l_evl: np.ndarray, in_place: bool = True):
        scala = 2
        x = (
            self.ham_diag.reshape(1, -1) - l_evl.reshape(-1, 1)
        ) * scala  # NOTE: s = I  so e_psi*s_diag = e_psi
        denom = 0.5 * (1 + x + np.sqrt(1 + (x - 1) * (x - 1))) / scala
        if in_place:
            np.divide(l_psi, denom, out=l_psi)
        else:
            return l_psi / denom
