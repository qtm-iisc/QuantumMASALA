r""" G+k Space Module

Like ``GSpace``, ``GkSpace`` also represents a Truncated Fourier Space, but
the vector in question is the  :math:`\mathbf{G} + \mathbf{k}` where
:math:`\mathbf{k}` is restricted to the first Brillouin Zone of the reciprocal
lattice. It is used for representing single-particle Bloch wavefunctions.

TODO : Complete documentation
"""

__all__ = ['GkSpace']

import numpy as np

from quantum_masala.core import ReciprocalLattice
from quantum_masala.core.fft import get_fft_driver
from .gspc import GSpace


class GkSpace:
    r"""Represents the space of :math:`\mathbf{G} + \mathbf{k}` vectors within
    the Kinetic Energy Cutoff

    Similar to ``GSpace``, but the vectors are shifted by a specified vector
    :math:`\mathbf{k}`. Required for representing bloch wavefunctions.

    Parameters
    ----------
    gspc : GSpace
        Describes the 'smooth' FFT grid.
    k_cryst : tuple[float, float, float]
        Vector :math:`\mathbf{k}` in crystal coordinates

    Notes
    -----
     Local Potentials operating on wavefunctions are required to be
     defined for this G-Space, otherwise the data has to be interpolated to
     this grid. This is requiure
    """
    def __init__(self, gspc: GSpace, k_cryst: tuple[float, float, float]):
        self.gspc = gspc
        """quantum_masala.core.Gspace : Represents the smooth FFT Grid; Local
        Potentials operating on wavefunctions shoud be required
        """
        self.k_cryst: np.ndarray = np.zeros(3, dtype='f8')
        """Crystal Coordinates of k-point
        """
        self.k_cryst[:] = k_cryst

        self.grid_shape = self.gspc.grid_shape
        self.reclat: ReciprocalLattice = self.gspc.recilat
        """Re
        """
        self.ecutwfc = self.gspc.ecut / 4

        gk_cryst = self.gspc.cryst.reshape((3, -1)) + self.k_cryst.reshape((3, 1))
        gk_cart = self.reclat.cryst2cart(gk_cryst)
        gk_2 = np.sum(gk_cart**2, axis=0)

        self.idxg = np.nonzero(gk_2 <= 2 * self.ecutwfc)[0]
        self.numgk = len(self.idxg)
        if self.numgk == 0:
            raise ValueError(f"Too few G-vectors within energy cutoff for "
                             f"'k_cryst'={self.k_cryst} and "
                             f"'ecut'={self.ecutwfc}"
                             )

        self.fft_mod = get_fft_driver()(self.gspc.grid_shape,
                                        tuple(arr[self.idxg] for arr in
                                              self.gspc.idxgrid),
                                        normalise_idft=False
                                        )

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
