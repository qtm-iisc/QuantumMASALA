# TODO: Refactor for spin-orbit calculatioons
from __future__ import annotations

from qtm.msg_format import shape_mismatch_msg

__all__ = ["KSHam"]
from collections.abc import Sequence
import numpy as np

from qtm.containers import FieldRType, get_WavefunG
from qtm.gspace import GkSpace
from qtm.pseudo import NonlocGenerator

from qtm.logger import qtmlogger
from qtm.msg_format import *


class KSHam:
    def __init__(
        self,
        gkspc: GkSpace,
        is_noncolin: bool,
        vloc: FieldRType,
        l_nloc: Sequence[NonlocGenerator],
    ):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(type_mismatch_msg("gkspc", gkspc, GkSpace))
        self.gkspc: GkSpace = gkspc
        self.ke_gk = get_WavefunG(gkspc, 1)((0.5 * self.gkspc.gk_norm2).astype("c16"))

        if not isinstance(is_noncolin, bool):
            raise TypeError(type_mismatch_msg("is_noncolin", is_noncolin, bool))
        self.is_noncolin: bool = is_noncolin

        if not isinstance(vloc, FieldRType):
            raise TypeError(type_mismatch_msg("vloc", vloc, FieldRType))
        if vloc.gspc is not gkspc.gwfn:
            raise ValueError(
                obj_mismatch_msg("vloc.gspc", vloc.gspc, "gkspc.gwfn", gkspc.gwfn)
            )

        if vloc.shape != (1 + is_noncolin,):
            if not is_noncolin and vloc.shape == ():
                pass
            else:
                raise ValueError(
                    value_mismatch_msg(
                        "vloc.shape",
                        vloc.shape,
                        f"{(1 + is_noncolin, )} when is_noncolin = {is_noncolin}",
                    )
                )
        self.vloc = vloc

        if not isinstance(l_nloc, Sequence) or any(
            not isinstance(nloc, NonlocGenerator) for nloc in l_nloc
        ):
            raise TypeError(type_mismatch_msg("l_nloc", l_nloc, NonlocGenerator))
        self.l_vkb_dij = []
        self.vnl_diag = 0
        for nloc in l_nloc:
            vkb, dij, vkb_diag = nloc.gen_vkb_dij(self.gkspc)
            self.l_vkb_dij.append((vkb, dij))
            self.vnl_diag += vkb_diag

    @qtmlogger.time("KSHam:h_psi")
    def h_psi(self, l_psi: get_WavefunG, l_hpsi: get_WavefunG):
        # l_hpsi[:] = self.ke_gk * l_psi
        assert l_psi.shape == l_hpsi.shape, shape_mismatch_msg(
            "l_psi", "l_hpsi", l_psi, l_hpsi
        )
        l_psi = l_psi.reshape(-1)
        l_hpsi = l_hpsi.reshape(-1)

        np.multiply(self.ke_gk, l_psi, out=l_hpsi)
        for psi, hpsi in zip(l_psi, l_hpsi):
            psi_r = psi.to_r()
            psi_r *= self.vloc.data.ravel()
            hpsi += psi_r.to_g()

        for vkb, dij in self.l_vkb_dij:
            # proj = vkb.vdot(l_psi)
            # l_hpsi += (dij @ proj).T @ vkb

            proj = vkb.vdot(l_psi)
            proj = dij @ proj
            l_hpsi.zgemm(vkb.data.T, proj.T, 0, 1, 1.0, l_hpsi.data.T, 1.0)
