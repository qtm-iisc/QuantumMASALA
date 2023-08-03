from __future__ import annotations
from qtm.config import NDArray
__all__ = ['GkSpace']

import numpy as np

from .gspc_base import GSpaceBase
from .gspc import GSpace
from .fft import FFT3DSticks

from qtm.constants import EPS


class GkSpace(GSpaceBase):

    FFT3D = FFT3DSticks

    def __init__(self, gwfn: GSpace, k_cryst: tuple[float, float, float], ecutwfn:float=None):
        if not isinstance(gwfn, GSpace):
            raise TypeError("'gwfn' must be a 'GSpace' instance. "
                            f"got '{type(gwfn)}'")
        self.gwfn = gwfn
        # self.ecutwfn = self.gwfn.ecut / 4

        # AS: Suggestion : Rename ecutwfc to ecut. ecutwfc makes sense in dft context, but in general, 
        #     it is just the cutoff for a shifted G-grid, which may correspond to anything.
        if ecutwfn==None:
            self.ecutwfn = self.gwfn.ecut / 4
        else:
            self.ecutwfn = ecutwfn
            

        self.k_cryst = tuple(k_cryst)
        gk_cryst = self.gwfn.g_cryst.copy().astype('f8')
        for ipol in range(3):
            gk_cryst[ipol] += self.k_cryst[ipol]
        gk_norm2 = self.gwfn.recilat.norm2(gk_cryst)
        self.idxgk = np.nonzero(gk_norm2 <= 2 * self.ecutwfn)[0]
        super().__init__(self.gwfn.recilat, self.gwfn.grid_shape,
                         self.gwfn.g_cryst[(slice(None), self.idxgk)],
                         )

    def create_buffer_gk(self, shape: tuple[int, ...], is_noncolin: bool) -> NDArray:
        if isinstance(shape, int):
            shape = (shape, )
        if not isinstance(is_noncolin, bool):
            raise TypeError("'is_noncolin' must be a boolean. "
                            f"got type {type(is_noncolin)}")
        return self.create_buffer((*shape, (1 + is_noncolin) * self.size_g))

    def check_buffer_gk(self, arr: NDArray, is_noncolin: bool):
        self.check_buffer(arr)
        if not isinstance(is_noncolin, bool):
            raise TypeError("'is_noncolin' must be a boolean. "
                            f"got type {type(is_noncolin)}")

        size_gk = (1 + is_noncolin) * self.size_g
        if not (arr.ndim >= 1 and arr.shape[-1] == size_gk):
            raise ValueError("shape of 'arr' invalid. "
                             f"got: arr.shape = {arr.shape}, "
                             f"is_noncolin = {is_noncolin}\n"
                             f"expected: arr.shape = {(..., size_gk)}"
                             )

    @property
    def gk_cryst(self) -> NDArray:
        gk_cryst = self.g_cryst.copy().astype('f8')
        for ipol in range(3):
            gk_cryst[ipol] += self.k_cryst[ipol]
        return gk_cryst

    @property
    def gk_cart(self) -> NDArray:
        """(``(size, )``, ``'f8'``) Cartesian coordinates of G+k vectors."""
        return self.recilat.cryst2cart(self.gk_cryst)

    @property
    def gk_tpiba(self) -> NDArray:
        r"""(``(3, size)``, ``'f8'``) Cartesian coordinates of G+k vectors in
        units of `tpiba` (:math:`\frac{2\pi}{a}`)."""
        return self.recilat.cryst2tpiba(self.gk_cryst)

    @property
    def gk_norm2(self) -> NDArray:
        """(``(size, )``, ``'f8'``) Norm squared of G+k vectors."""
        return self.recilat.norm2(self.gk_cryst, 'cryst')

    @property
    def gk_norm(self) -> NDArray:
        """(``(size, )``, ``'f8'``) Norm of G+k vectors."""
        return np.sqrt(self.gk_norm2)

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if not isinstance(other, type(self)):
            return False
        if self.gwfn != other.gwfn:
            return False
        return np.sqrt(sum(
                (self.k_cryst[ipol] - other.k_cryst[ipol])**2 for ipol in range(3)
        )) < EPS
