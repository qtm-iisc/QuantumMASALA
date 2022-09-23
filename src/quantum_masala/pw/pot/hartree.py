import numpy as np

from quantum_masala.core import Rho

from .base import LocalPot


class Hartree(LocalPot):
    def __init__(
        self, rho: Rho
    ):
        super().__init__(rho)
        self._g = np.empty(self.grho.numg, dtype="c16")
        self._r = np.empty(self.grho.grid_shape, dtype="c16")
        self._en = None

    @property
    def g(self):
        return self._g

    @property
    def r(self):
        return self._r

    @property
    def en(self):
        return self._en

    def sync(self):
        self._g = self.pwcomm.world_comm.Bcast(self._g)
        self._r[:] = self.fft_rho.g2r(self._g, self._r)

    def compute(self):
        rho_gcut_unpol = np.sum(self.rho.g, axis=0)
        self._g[0] = 0
        self._g[1:] = 4 * np.pi * rho_gcut_unpol[1:] / self.grho.norm2[1:]
        self._r[:] = self.fft_rho.g2r(self._g, self._r)

        self._en = 0.5 * self.rho.integral_rho_f_dv(self.r)
        self.sync()
