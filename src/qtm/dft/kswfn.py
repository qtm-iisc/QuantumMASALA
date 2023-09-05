from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
__all__ = ['KSWfn']

import numpy as np

from qtm.gspace import GkSpace
from qtm.containers import WavefunGType, get_WavefunG, FieldRType
from qtm.constants import TPIJ

from qtm.config import CUPY_INSTALLED, qtmconfig, NDArray


def get_rng_module(arr: NDArray):
    """Returns the ``random`` submodule corresponding the type of the
    input array
    """
    if isinstance(arr, np.ndarray):
        return np.random
    if CUPY_INSTALLED:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.random
    else:
        raise NotImplementedError(f"'{type(arr)}' not recognized/supported.")


class KSWfn:
    """Container for storing information about the eigenstates of the Kohn-Sham
    Hamiltonian

    Contains the KS eigen-wavefunctions, its corresponding eigenvalues and
    occupation number.

    Parameters
    ----------
    gkspc : GkSpace
        Represents the basis of the single-particle Bloch wavefunctions at
        k-point `gkspc.k_cryst`
    k_weight : float
        Weight associated to the k-point represented by `gkspc`
    numbnd : int
        Number of KS bands stored
    is_noncolin : bool
        If True, wavefunctions are spin-dependent. For non-colinear calculations

    """
    def __init__(self, gkspc: GkSpace,
                 k_weight: float, numbnd: int, is_noncolin: bool):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(f"'gkspc' must be a {GkSpace} instance. "
                            f"got {type(gkspc)}.")

        self.gkspc: GkSpace = gkspc
        """Represents the basis of the single-particle Bloch wavefunction at
        k-point `gkspc.k_cryst`
        """
        self.k_cryst: tuple[float, float, float] = self.gkspc.k_cryst
        """k-point in crystal coordinates corresponding to `gkspc`
        """

        self.k_weight = float(k_weight)
        """Weight associated to the k-point represented by `gkspc`
        """

        if not isinstance(numbnd, int) or numbnd <= 0:
            raise TypeError(f"'numbnd' must be a positive integer. "
                            f"got {numbnd} (type {type(numbnd)}).")
        self.numbnd: int = numbnd
        """Number of KS bands stored
        """

        if not isinstance(is_noncolin, bool):
            raise TypeError(f"'is_noncolin' must be a boolean. "
                            f"got '{type(is_noncolin)}'.")
        self.is_noncolin: bool = is_noncolin
        """If True, wavefunctions are spin-dependent.
        For non-colinear calculations
        """

        WavefunG = get_WavefunG(self.gkspc, 1 + self.is_noncolin)
        self.evc_gk: WavefunGType = WavefunG.empty(self.numbnd)
        """Contains the KS eigen-wavefunctions
        """
        self.evl: NDArray = np.empty(self.numbnd, dtype='f8',
                                     like=self.evc_gk.data)
        """Eigenvalues of the eigenkets in `evc_gk`
        """
        self.occ: NDArray = np.empty(self.numbnd, dtype='f8',
                                     like=self.evc_gk.data)
        """Occupation number of the eigenkets in `evc_gk`
        """

    def init_random(self):
        """Initializes `evc_gk` with an unnormalized randomized
        wavefunction"""
        rng_mod = get_rng_module(self.evc_gk.data)
        seed_k = np.array(self.k_cryst).view('uint')
        rng = rng_mod.default_rng([seed_k, qtmconfig.rng_seed])
        data = self.evc_gk.data
        rng.random(out=data.view('f8'))
        np.multiply(data.real, np.exp(TPIJ * data.imag), out=data)
        self.evc_gk /= 1 + self.gkspc.gk_norm2

    def compute_rho(self, ibnd: slice | Sequence[int] = slice(None)) -> FieldRType:
        """Constructs a density from the eigenkets `evc_gk` and occupation
        `occ`"""
        self.evc_gk[ibnd].normalize()
        rho: FieldRType = sum(
            occ * wfn.to_r().get_density(normalize=False)
            for wfn, occ in zip(self.evc_gk[ibnd], self.occ[ibnd])
        )
        rho /= rho.gspc.reallat_cellvol
        return rho if self.is_noncolin else rho[0]
