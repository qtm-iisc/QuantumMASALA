from __future__ import annotations
__all__ = ['KSHam']
from collections.abc import Sequence
import numpy as np

from qtm.containers import FieldR, WavefunG
from qtm.gspace import GkSpace
from qtm.pseudo import NonlocGenerator


class KSHam:

    def __init__(self, gkspc: GkSpace, vloc: FieldR,
                 l_nloc: Sequence[NonlocGenerator]):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(f"'gkspc' must be a {GkSpace} instance. "
                            f"got '{type(gkspc)}'.")
        self.gkspc: GkSpace = gkspc
        self.ke_gk = WavefunG(gkspc, (0.5 * self.gkspc.gk_norm2).astype('c16'))

        if not isinstance(vloc, FieldR):
            raise TypeError(f"'vloc' must be a {FieldR} instance. "
                            f"got '{type(vloc)}'.")
        if vloc.gspc is not self.gkspc.gwfn:
            raise ValueError("mismatch between 'vloc.gspc' and 'gkspc.gwfn'")
        if vloc.shape != ():
            raise ValueError("'vloc' must be a scalar Field i.e. shape =  (). "
                             f"got vloc.shape = {vloc.shape}")
        self.vloc = vloc

        if not isinstance(l_nloc, Sequence) or any(
                not isinstance(nloc, NonlocGenerator) for nloc in l_nloc
        ):
            raise TypeError(
                f"'l_nloc' must be a sequence of '{NonlocGenerator}' instances. "
            )

        self.l_vkb_dij = []
        self.vnl_diag = 0
        for nloc in l_nloc:
            vkb, dij, vkb_diag = nloc.gen_vkb_dij(self.gkspc)
            self.l_vkb_dij.append((vkb, dij))
            self.vnl_diag += vkb_diag

    def h_psi(self, l_psi: WavefunG, l_hpsi: WavefunG):
        # l_hpsi[:] = self.ke_gk * l_psi
        assert l_psi.shape == l_hpsi.shape
        l_psi = l_psi.reshape(-1)
        l_hpsi = l_hpsi.reshape(-1)

        np.multiply(self.ke_gk, l_psi, out=l_hpsi)
        for psi, hpsi in zip(l_psi, l_hpsi):
            psi_r = psi.to_r()
            psi_r *= self.vloc.data
            hpsi += psi_r.to_g()

        for vkb, dij in self.l_vkb_dij:
            # proj = vkb.vdot(l_psi)
            # l_hpsi += (dij @ proj).T @ vkb

            proj = vkb.vdot(l_psi)
            proj = dij @ proj

            l_hpsi._zgemm(vkb.data.T, proj.T,
                          0, 1, 1.0, l_hpsi.data.T, 1.0)
