import numpy as np

from .gspc import GSpace


class GSpaceWfc:

    def __init__(self, gspc: GSpace, k_cryst: np.ndarray):
        self.gspc = gspc

        self.k_cryst = np.zeros(3, dtype='f8')
        self.k_cryst[:] = k_cryst

        self.reclat = self.gspc.recilat
        self.ecutwfc = self.gspc.ecut / 4

        gk_cryst = self.gspc.cryst.reshape((3, -1)) + self.k_cryst.reshape((3, 1))
        gk_cart = self.reclat.cryst2cart(gk_cryst)
        gk_2 = np.sum(gk_cart**2, axis=0)

        self.idxg = np.nonzero(gk_2 <= 2 * self.ecutwfc)[0]
        self.numgk = len(self.idxg)
        if self.numgk == 0:
            raise ValueError(f"Too few G-vectors within energy cutoff for 'k_cryst'={self.k_cryst} and "
                             f"'ecut'={self.ecutwfc}")

    @property
    def idxgrid(self):
        return tuple(arr[self.idxg] for arr in self.gspc.idxgrid)

    @property
    def cryst(self):
        return self.gspc.cryst[:, self.idxg] + self.k_cryst.reshape(3, 1)

    @property
    def cart(self):
        return self.reclat.cryst2cart(self.cryst)

    @property
    def tpiba(self):
        return self.reclat.cryst2tpiba(self.cryst)

    @property
    def g_cryst(self):
        return self.gspc.cryst[:, self.idxg]

    @property
    def g_cart(self):
        return self.reclat.cryst2cart(self.g_cryst)

    @property
    def g_tpiba(self):
        return self.reclat.cryst2tpiba(self.g_cryst)

    @property
    def norm2(self):
        return np.sum(self.cart ** 2, axis=0)

    @property
    def norm(self):
        return np.linalg.norm(self.cart, axis=0)
