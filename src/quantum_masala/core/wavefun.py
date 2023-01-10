__all__ = ['Wavefun']

import numpy as np

from quantum_masala.core import GSpace, GkSpace


class Wavefun:
    """Container for Bloch Wavefunctions representing single-particle states of
    non-interacting systems.

    Parameters
    ----------
    gspc : GSpace
        G-Space representing the smooth grid for wavefunctions. Note that its cutoff
        ``ecut`` must be 4X the wavefunction Kinetic Energy cutoff ``ecutwfc``.
    k_cryst : tuple[float, float, float]
        Crystal coordinates of k-point.
    numbnd : int
        Number of bands.
    is_spin : bool
        Spin-polarization if ``True``.
    is_noncolin : bool
        Non-collinear calculation if ``True``. To be implemented.
    """
    def __init__(self, gspc: GSpace, k_cryst: tuple[float, float, float],
                 numbnd: int, is_spin: bool, is_noncolin: bool):
        self.k_cryst: np.ndarray = np.empty(3, dtype='f8')
        self.k_cryst[:] = k_cryst
        """(``(3, )``, ``'f8'``) Crystal Coordinates of k-point
        """

        self.gspc: GSpace = gspc
        """Represents the smooth FFT grid for wavefunctions
        """
        self.gkspc: GkSpace = GkSpace(self.gspc, tuple(self.k_cryst))
        r"""Represents the set of :math:`\mathbf{G}+\mathbf{k}` vectors within
        Kinetic energy cutoff ``ecutwfc = self.gspc.ecut / 4``
        """

        if not isinstance(numbnd, int) or numbnd < 1:
            raise ValueError("'numbnd' must be a positive integer. "
                             f"got '{numbnd}' (type {type(numbnd)})")
        self.numbnd: int = numbnd
        """Number of bands
        """

        if not isinstance(is_spin, bool):
            raise ValueError("'is_spin' must be a boolean. "
                             f"got {type(is_spin)}")
        self.is_spin: bool = is_spin
        """Spin-polarization if ``True``
        """

        if not isinstance(is_noncolin, bool):
            raise ValueError("'is_noncolin' must be a boolean. "
                             f"Got {type(is_noncolin)}")
        if not is_spin and is_noncolin:
            raise ValueError("'is_spin' must be 'True' for non-collinear calculation.")
        if is_noncolin:
            raise ValueError("'is_noncolin = True' yet to be implemented")
        self.is_noncolin: bool = is_noncolin
        """Non-collinear if ``True``
        """

        self.numspin: int = 1 + self.is_spin
        """``1 + self.is_spin``
        """
        self.evc_gk: np.ndarray = np.empty((1 + self.is_spin, self.numbnd,
                                            self.gkspc.numgk*(1 + self.is_noncolin)),
                                           dtype='c16')
        """(``(1+self.is_spin, self.numbnd, self.gkspc.numgk*(1+self.is_noncolin))``,
        ``'c16'``) List of wavefunctions in PW Basis described by ``self.gkspc``
        """

        self.occ: np.ndarray = np.empty((1 + self.is_spin, self.numbnd), dtype='f8')
        """(``(1+self.is_spin, self.numbnd)``, ``'f8'``) List of occupation
        numbers
        """

    def normalize(self):
        self.evc_gk /= np.linalg.norm(self.evc_gk, axis=-1, keepdims=True) \
                       * np.sqrt(self.gspc.reallat_cellvol) \
                       / np.prod(self.gspc.grid_shape)

    def get_amp2_r(self, sl) -> np.ndarray:
        shape = self.evc_gk[sl].shape
        if len(shape) < 1 or shape[-1] != self.gkspc.numgk:
            raise IndexError("invalid index. possibly too many indices specified.")
        evc_r = self.gkspc.fft_mod.g2r(self.evc_gk[sl])
        l_amp_r = evc_r.conj() * evc_r
        return l_amp_r
