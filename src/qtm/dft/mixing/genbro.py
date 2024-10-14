__all__ = ['GenBroyden']


import numpy as np

from qtm.containers import FieldGType
from qtm.dft import DFTCommMod

from .base import MixModBase


class GenBroyden(MixModBase):

    def __init__(self, dftcomm: DFTCommMod, rho: FieldGType,
                 beta: float, mixdim: int):
        super().__init__(dftcomm, rho, beta, mixdim)
        self.idxiter = 0

        if self.is_root_pwgrp:
            self.rho_old = self.FieldG.empty(self.numspin)
            self.res_old = self.FieldG.empty(self.numspin)

            self.l_del_rho = self.FieldG.empty((self.mixdim, self.numspin))
            self.l_del_res = self.FieldG.empty((self.mixdim, self.numspin))
            self.overlap = np.empty((self.mixdim, self.mixdim), dtype='c16',
                                    like=self.grho.g_cryst)
        else:
            self.rho_old, self.res_old = None, None
            self.l_del_rho, self.l_del_res, self.overlap = None, None, None

        shape = (self.numspin, self.grho.size_g)
        self.G_1 = -self.beta * np.ones(shape)


    def _mix(self, rho_in: FieldGType, rho_out: FieldGType) -> FieldGType:
        res = rho_out.data - rho_in.data

        numdim = min(self.idxiter, self.mixdim)
        if numdim == 0:
            del_rho = self.beta * res
        else:
            isave = (self.idxiter - 1) % self.mixdim
            self.l_del_rho[isave] = rho_in.data - self.rho_old
            self.l_del_res[isave] = res - self.res_old
            for i in range(numdim):
                self.overlap[isave, i] = self._dot(
                    self.l_del_res[isave], self.l_del_res[i]
                )
                self.overlap[i, isave] = self.overlap[isave, i]

            try:
                overlap_inv = np.linalg.inv(self.overlap[:numdim, :numdim])
            except np.linalg.LinAlgError as e:
                print(
                    "'error in charge mixing routine: cannot invert overlap matrix. "
                    "Try using a different mixing method."
                )
                raise np.linalg.LinAlgError(e)

            del_rho = -self.G_1 * res
            l_dot = np.empty(numdim, dtype="c16")
            for i in range(numdim):
                l_dot[i] = self._dot(self.l_del_res[i], -res)

            comp = overlap_inv @ l_dot
            for i in range(numdim):
                del_rho += comp[i] * (self.l_del_rho[i] - self.G_1 * self.l_del_res[i])

        self.rho_old[:] = rho_in.data
        self.res_old[:] = res
        self.idxiter += 1

        rho_in._data += del_rho
        return rho_in