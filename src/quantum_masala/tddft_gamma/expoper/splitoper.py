__all__ = ['SplitOper']
import numpy as np
from scipy.linalg import expm, block_diag

from quantum_masala.core import GkSpace, RField
from quantum_masala.pseudo import NonlocGenerator

from .base import TDExpOperBase


class SplitOper(TDExpOperBase):

    __slots__ = ['l_exp_dij']

    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: RField, l_nloc: list[NonlocGenerator],
                 time_step: float):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc,
                         time_step)

        # self.numkb = len(self.l_vkb_dij)
        # self.l_exp_dij = []
        # for vkb, dij in self.l_vkb_dij:
        #     ovl = vkb.conj() @ vkb.T
        #     self.l_exp_dij.append(
        #         np.linalg.inv(ovl) @ (expm(-0.5j * self.time_step * (ovl @ dij))
        #                               - np.identity(dij.shape[0]))
        #     )

        self.vkb_all = np.concatenate([vkb for vkb, _ in self.l_vkb_dij], axis=0)
        self.dij_all = block_diag(*[dij for _, dij in self.l_vkb_dij])
        ovl = self.vkb_all.conj() @ self.vkb_all.T
        self.exp_dij_all = np.linalg.inv(ovl) @ (
            expm(-0.5j * self.time_step * (ovl @ self.dij_all))
            - np.identity(self.dij_all.shape[0])
        )

        self.oper_ke_gk = np.exp(-0.5j * self.time_step * self.ke_gk)
        self.oper_vloc_r = None

    def update_vloc(self, vloc: RField):
        self._check_vloc(vloc)
        self.oper_vloc_r = np.exp(-1j * self.time_step * vloc.r)

    def oper_ke(self, l_prop_psi: np.ndarray):
        np.multiply(l_prop_psi, self.oper_ke_gk, out=l_prop_psi)

    def oper_nl(self, l_prop_psi: np.ndarray, reverse: bool):
        # l_ikb = range(self.numkb)
        # if reverse:
        #     l_ikb = reversed(l_ikb)
        # for ikb in l_ikb:
        #     vkb, dij = self.l_vkb_dij[ikb]
        #     proj = vkb.conj() @ l_prop_psi.transpose((0, 2, 1))
        #     l_prop_psi += (self.l_exp_dij[ikb] @ proj).transpose((0, 2, 1)) @ vkb

        proj = self.vkb_all.conj() @ l_prop_psi.transpose((0, 2, 1))
        l_prop_psi += (self.exp_dij_all @ proj).transpose((0, 2, 1)) @ self.vkb_all

    def oper_vloc(self, l_prop_psi: np.ndarray):
        self.gkspc.fft_mod.r2g(self.oper_vloc_r * self.gkspc.fft_mod.g2r(l_prop_psi), l_prop_psi)

    def prop_psi(self, l_psi: np.ndarray, l_prop_psi: np.ndarray):
        l_prop_psi[:] = l_psi
        self.oper_ke(l_prop_psi)
        self.oper_nl(l_prop_psi, False)
        self.oper_vloc(l_prop_psi)
        self.oper_nl(l_prop_psi, True)
        self.oper_ke(l_prop_psi)
