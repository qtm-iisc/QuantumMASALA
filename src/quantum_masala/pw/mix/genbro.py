import numpy as np

from quantum_masala.core import Rho, Wfn2Rho

from .base import MixMod


class GenBroyden(MixMod):
    def __init__(self, rho: Rho, wfn2rho: Wfn2Rho,
                 beta: float, mixdim: int):
        super().__init__(rho, wfn2rho, beta, mixdim)

        self.idxiter = 0

        rhoshape = (rho.numspin, self.grho.numg)

        self.rho_old = np.empty(rhoshape, dtype="c16")
        self.rho_old[:] = rho.g
        self.res_old = np.empty(rhoshape, dtype="c16")

        self.l_del_rho = np.empty((self.mixdim, *rhoshape), dtype="c16")
        self.l_del_res = np.empty((self.mixdim, *rhoshape), dtype="c16")
        self.overlap = np.empty((self.mixdim, self.mixdim), dtype="c16")

        self.G_1 = -self.beta * np.ones(rhoshape)

    def _mix(self, rho_in: Rho, rho_out: Wfn2Rho) -> None:
        res = rho_out.g - rho_in.g

        numdim = min(self.idxiter, self.mixdim)
        if numdim == 0:
            del_rho = self.beta * res
        else:
            isave = (self.idxiter - 1) % self.mixdim
            self.l_del_rho[isave] = rho_in.g - self.rho_old
            self.l_del_res[isave] = res - self.res_old
            for i in range(numdim):
                self.overlap[isave, i] = self.dot(
                    self.l_del_res[isave], self.l_del_res[i]
                )
                self.overlap[i, isave] = self.overlap[isave, i]

            try:
                overlap_inv = np.linalg.inv(self.overlap[:numdim, :numdim])
            except np.linalg.LinAlgError as e:
                print(
                    "'error in charge mixing routine: cannot invert overlap matrix. "
                    "Use a different mixing method."
                )
                raise np.linalg.LinAlgError(e)

            del_rho = -self.G_1 * res
            l_dot = np.empty(numdim, dtype="c16")
            for i in range(numdim):
                l_dot[i] = self.dot(self.l_del_res[i], -res)

            comp = overlap_inv @ l_dot
            for i in range(numdim):
                del_rho += comp[i] * (self.l_del_rho[i] - self.G_1 * self.l_del_res[i])

        self.rho_old[:] = rho_in.g
        self.res_old[:] = res
        self.idxiter += 1

        rho_in.update(rho_in.g + del_rho)
