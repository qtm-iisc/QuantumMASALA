from __future__ import annotations

from typing import Sequence
__all__ = ['GkSpace']

import numpy as np

from .base import GSpaceBase
from .gspc import GSpace
from qtm.fft import FFT3DFull

from qtm.config import NDArray


class GkSpace(GSpaceBase):

    FFT3D = FFT3DFull
    _normalise_idft = False

    def __init__(self, gwfn: GSpace, k_cryst: tuple[float, float, float], ecutwfn:float=None):
        if not isinstance(gwfn, GSpace):
            raise TypeError("'gwfn' must be a 'GSpace' instance. "
                            f"got '{type(gwfn)}'")
        self.gwfn = gwfn

        if ecutwfn==None:
            self.ecutwfn = self.gwfn.ecut / 4
        else:
            self.ecutwfn = ecutwfn

        self.k_cryst = tuple(k_cryst)
        g_cryst = self.gwfn.g_cryst[:, self.gwfn.idxsort]
        gk_cryst = g_cryst.astype('f8')
        for ipol in range(3):
            gk_cryst[ipol] += self.k_cryst[ipol]
        gk_norm2 = self.gwfn.recilat.norm2(gk_cryst)
        self.idxgk = np.nonzero(gk_norm2 <= 2 * self.ecutwfn)[0]
        super().__init__(self.gwfn.recilat, self.gwfn.grid_shape,
                         g_cryst[:, self.idxgk],
                         )

        self.gk_cryst = self.g_cryst.copy().astype('f8')
        for ipol in range(3):
            self.gk_cryst[ipol] += self.k_cryst[ipol]

    @property
    def gk_cart(self) -> NDArray:
        return self.recilat.cryst2cart(self.gk_cryst)

    @property
    def gk_tpiba(self) -> NDArray:
        r"""(``(3, size)``, ``'f8'``) Cartesian coordinates of G+k vectors in
        units of `tpiba` (:math:`\frac{2\pi}{a}`)."""
        return self.recilat.cryst2tpiba(self.gk_cryst)

    @property
    def gk_norm2(self) -> NDArray:
        # TODO: Bug when using 'cryst'. Fix it
        return self.recilat.norm2(self.gk_cart, 'cart')

    @property
    def gk_norm(self) -> NDArray:
        """(``(size, )``, ``'f8'``) Norm of G+k vectors."""
        return np.sqrt(self.gk_norm2)

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
        return np.sum(np.square(self.recilat.recvec @ l_vecs), axis=0)