__all__ = ['ModBroyden']

import numpy as np

from quantum_masala.core import GField
from .base import MixModBase


class ModBroyden(MixModBase):
    def __init__(self, rho: GField,
                 beta: float, mixdim: int):
        super().__init__(rho, beta, mixdim)

        self.idxiter = 0

        shape = (self.numspin, self.grho.numg)

        self.rho_old = np.empty(shape, dtype="c16")
        self.res_old = np.empty(shape, dtype="c16")

        self.l_del_rho = np.empty((self.mixdim, *shape), dtype="c16")
        self.l_del_res = np.empty((self.mixdim, *shape), dtype="c16")
        self.overlap = np.empty((self.mixdim, self.mixdim), dtype="c16")

    def _mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
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
                print(
                    "'error in charge mixing routine: cannot invert overlap matrix. "
                    "Try using a different mixing method."
                )
                raise np.linalg.LinAlgError(e)

            for i in range(numdim):
                overlap_inv[:i, i] = overlap_inv[i, :i]

            l_dot = np.empty(numdim, dtype="c16")
            for i in range(numdim):
                l_dot[i] = self._dot(self.l_del_res[i], res)

            comp = overlap_inv.T @ l_dot
            for i in range(numdim):
                rho_in -= comp[i] * self.l_del_rho[i]
                res -= comp[i] * self.l_del_res[i]

        self.idxiter += 1

        rho_in += self.beta * res
        return rho_in
