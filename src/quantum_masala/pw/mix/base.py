from abc import ABC, abstractmethod
import numpy as np

from quantum_masala.core import PWComm, GSpace, Rho, Wfn2Rho


class MixMod(ABC):
    pwcomm: PWComm()
    grho: GSpace
    beta: float
    mixdim: int

    @abstractmethod
    def __init__(self, rho: Rho, wfn2rho: Wfn2Rho,
                 beta: float, mixdim: int):
        if rho.grho.grid_shape != wfn2rho.grho.grid_shape:
            raise NotImplementedError("non-identical hard-grid and soft-grid not implemented.")
        self.grho_in = rho.grho
        self.grho_out = wfn2rho.grho
        self.grho = self.grho_in
        self.numspin = rho.numspin

        self.beta = beta
        self.mixdim = mixdim

    def dot(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        chden1 = np.sum(rho1, axis=0)
        chden2 = np.sum(rho2, axis=0)

        fac = (
            0.5 * 4 * np.pi * self.grho.reallat_dv / np.prod(self.grho.grid_shape)
        )  # TODO: Check the denominator
        dot = fac * np.sum((chden1.conj() * chden2).real[1:] / self.grho.norm2[1:])
        if self.numspin == 1:
            dot *= 4
        elif self.numspin == 2:
            spden1 = rho1[0] - rho1[1]
            spden2 = rho2[0] - rho2[1]
            tpiba = self.grho.recilat.tpiba
            dot += fac * np.sum(spden1.conj() * spden2).real / tpiba**2
        return dot

    def mix(self, rho_in: Rho, rho_out: Wfn2Rho) -> Rho:
        self._mix(rho_in, rho_out)
        return rho_in

    def compute_error(self, rho_in: Rho, rho_out: Wfn2Rho) -> float:
        res = rho_in.g - rho_out.g
        return self.dot(res, res)

    @abstractmethod
    def _mix(self, rho_in: Rho, rho_out: Wfn2Rho) -> None:
        pass
