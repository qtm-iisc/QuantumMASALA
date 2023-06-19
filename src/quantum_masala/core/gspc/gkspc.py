r""" G+k Space Module

Like ``GSpace``, ``GkSpace`` also represents a Truncated Fourier Space, but
the vector in question is the  :math:`\mathbf{G} + \mathbf{k}` where
:math:`\mathbf{k}` is restricted to the first Brillouin Zone of the reciprocal
lattice. It is used for representing single-particle Bloch wavefunctions.

TODO : Complete documentation
"""

__all__ = ['GkSpace']

from typing import Sequence
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
    def __init__(self, gspc: GSpace, k_cryst: tuple[float, float, float], ecutwfc:float=None):
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
        
        # AS: Suggestion : Rename ecutwfc to ecut. ecutwfc makes sense in dft context, but in general, 
        #     it is just the cutoff for a shifted G-grid, which may correspond to anything.
        if ecutwfc==None:
            self.ecutwfc = self.gspc.ecut / 4
        else:
            self.ecutwfc = ecutwfc
            
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
    
    @property
    def g_norm2(self):
        return np.sum(self.g_cart ** 2, axis=0)

    @property
    def gk_indices_tosorted(self):
        """Copy of ``sort_cryst_like_BGW``"""
        from quantum_masala.gw.core import sort_cryst_like_BGW
        # return sort_cryst_like_BGW(self.cryst, self.norm)

        # Sorting order same as BerkeleyGW
        cryst = self.cryst
        key_array = self.norm2
        indices_cryst_sorted = sort_cryst_like_BGW(cryst, key_array)
        return indices_cryst_sorted
    
    @property
    def gk_indices_fromsorted(self):
        return np.argsort(self.gk_indices_tosorted)

    def cryst_to_norm2(self, l_vecs: Sequence) -> Sequence:
        """Calculate the norm^2 of a given list of vectors in crystal coordinates.
        
        Parameters
        ----------
        l_vec: Sequence
            List of vectors in crystal coordinates. shape: (3,:)
        
        Returns
        -------
        np.ndarray of shape (:)
        """
        return np.sum(np.square(self.reclat.recvec @ l_vecs), axis=0)