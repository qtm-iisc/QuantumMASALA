__all__ = ['MixModBase']

from abc import ABC, abstractmethod
import numpy as np

from quantum_masala.core import GSpace, GField, rho_check
from quantum_masala import config


class MixModBase(ABC):

    @abstractmethod
    def __init__(self, rho: GField, beta: float, mixdim: int):
        self.pwcomm = config.pwcomm

        rho_check(rho)
        self.grho = rho.gspc
        self.numspin = rho.shape[0]

        self.beta = beta
        self.mixdim = mixdim

    def _dot(self, rho1_g: np.ndarray, rho2_g: np.ndarray) -> float:
        chden1 = np.sum(rho1_g, axis=0)
        chden2 = np.sum(rho2_g, axis=0)

        fac = (
            0.5 * 4 * np.pi * self.grho.reallat_dv / np.prod(self.grho.grid_shape)
        )
        dot = fac * np.sum((chden1.conj() * chden2).real[1:] / self.grho.norm2[1:])
        if self.numspin == 2:
            spden1 = rho1_g[0] - rho1_g[1]
            spden2 = rho2_g[0] - rho2_g[1]
            tpiba = self.grho.recilat.tpiba
            dot += fac * np.sum(spden1.conj() * spden2).real / tpiba**2
        return dot

    def mix(self, rho_in: GField, rho_out: GField) -> GField:
        rho_next_g = self._mix(rho_in.g, rho_out.g)
        return GField.from_array(self.grho, rho_next_g)

    def compute_error(self, rho_in: GField, rho_out: GField) -> float:
        res = rho_in.g - rho_out.g
        return self._dot(res, res)

    @abstractmethod
    def _mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        pass
