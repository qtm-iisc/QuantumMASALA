__all__ = ["ModBroyden"]

import numpy as np

from qtm.containers import FieldGType
from qtm.dft import DFTCommMod

from .base import MixModBase


class ModBroyden(MixModBase):
    def __init__(self, dftcomm: DFTCommMod, rho: FieldGType, beta: float, mixdim: int):
        super().__init__(dftcomm, rho, beta, mixdim)
        self.idxiter = 0

        if self.is_root_pwgrp:
            self.rho_old = self.FieldG.empty(self.numspin)
            self.res_old = self.FieldG.empty(self.numspin)

            self.l_del_rho = self.FieldG.empty((self.mixdim, self.numspin))
            self.l_del_res = self.FieldG.empty((self.mixdim, self.numspin))
            self.overlap = np.empty(
                (self.mixdim, self.mixdim), dtype="c16", like=self.grho.g_cryst
            )
        else:
            self.rho_old, self.res_old = None, None
            self.l_del_rho, self.l_del_res, self.overlap = None, None, None

    def _mix(self, rho_in: FieldGType, rho_out: FieldGType) -> FieldGType:
        self._check_rho(rho_in, rho_out)
        res = rho_out - rho_in

        numdim = min(self.idxiter, self.mixdim)
        if numdim == 0:
            self.rho_old[:] = rho_in
            self.res_old[:] = res
        else:
            isave = (self.idxiter - 1) % self.mixdim
            self.l_del_rho[isave] = self.rho_old - rho_in
            self.l_del_res[isave] = self.res_old - res
            self.rho_old[:] = rho_in
            self.res_old[:] = res
            for i in range(numdim):
                self.overlap[isave, i] = self._dot(
                    self.l_del_res[i], self.l_del_res[isave]
                )
                self.overlap[i, isave] = self.overlap[isave, i]

            try:
                overlap_inv = np.linalg.inv(self.overlap[:numdim, :numdim])
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    "'error in charge mixing routine: cannot invert overlap matrix. "
                    "Try using a different mixing method."
                ) from e

            for i in range(numdim):
                overlap_inv[:i, i] = overlap_inv[i, :i]

            l_dot = np.empty(numdim, dtype="c16", like=self.grho.g_cryst)
            for i in range(numdim):
                l_dot[i] = self._dot(self.l_del_res[i], res)

            comp = overlap_inv.T @ l_dot
            rho_in = rho_in.copy()
            for i in range(numdim):
                rho_in -= comp[i] * self.l_del_rho[i]
                res -= comp[i] * self.l_del_res[i]

        self.idxiter += 1

        rho_in += self.beta * res
        return rho_in
