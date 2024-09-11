__all__ = ['SplitOper']
import numpy as np
from qtm.containers.field import FieldRType
from qtm.containers.wavefun import get_WavefunG
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.pseudo.nloc import NonlocGenerator
from scipy.linalg import expm, block_diag

from .base import TDExpOperBase


class SplitOper(TDExpOperBase):

    __slots__ = ['l_exp_dij_halfstep']

    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: FieldRType, l_nloc: list[NonlocGenerator],
                 time_step: float):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc,
                         time_step)

        self.l_vkb_dij = []
        self.vnl_diag = 0
        for nloc in l_nloc:
            vkb, dij, vkb_diag = nloc.gen_vkb_dij(self.gkspc)
            print(type(vkb))
            self.l_vkb_dij.append((vkb, dij))
            self.vnl_diag += vkb_diag

        self.l_exp_dij_halfstep = []
        for vkb, dij in self.l_vkb_dij:
            ovl = vkb.vdot(vkb)
            self.l_exp_dij_halfstep.append(
                np.linalg.inv(ovl) @ (expm(-0.5j * self.time_step * (ovl @ dij))
                                      - np.identity(dij.shape[0]))
            )


        self.oper_exp_ke_gk_halfstep = np.exp(-0.5j * self.time_step * self.ke_gk)
        self.oper_exp_vloc_r_fullstep = None

    def update_vloc(self, vloc: FieldRType):
        # self._check_vloc(vloc)
        fac = -1j * self.time_step * np.prod(self.gkspc.grid_shape)
        # The line below must contain a factor 1/ np.prod(self.gkspc.grid_shape)
        # But since the wfn's are normalized at each step it is skipped
        self.oper_exp_vloc_r_fullstep = np.exp(fac * vloc.data.ravel())

    def oper_ke(self, l_prop_psi: list[KSWfn]):
        np.multiply(self.oper_exp_ke_gk_halfstep, l_prop_psi[0].evc_gk, out=l_prop_psi[0].evc_gk)

    def oper_nl(self, l_prop_psi: np.ndarray, reverse: bool):
        l_ikb = range(len(self.l_vkb_dij))
        if reverse:
            l_ikb = reversed(l_ikb)
        for ikb in l_ikb:
            vkb, dij = self.l_vkb_dij[ikb]
            proj = vkb.vdot(l_prop_psi[0].evc_gk)
            l_prop_psi[0].evc_gk += (self.l_exp_dij_halfstep[ikb] @ proj).T @ vkb

    def oper_vloc(self, l_prop_psi: np.ndarray):
        psi_r = l_prop_psi[0].evc_gk.to_r()
        psi_r *= self.oper_exp_vloc_r_fullstep
        l_prop_psi[0].evc_gk[:] = psi_r.to_g()[:]

    def prop_psi(self, l_psi: np.ndarray, l_prop_psi: np.ndarray):
        l_prop_psi[:] = l_psi
        self.oper_ke(l_prop_psi)
        self.oper_nl(l_prop_psi, False)
        self.oper_vloc(l_prop_psi)
        self.oper_nl(l_prop_psi, True)
        self.oper_ke(l_prop_psi)
