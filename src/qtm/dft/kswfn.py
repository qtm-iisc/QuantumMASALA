from __future__ import annotations
from typing import TYPE_CHECKING, List, Union

from qtm.logger import COMM_WORLD

if TYPE_CHECKING:
    from collections.abc import Sequence
__all__ = ["KSWfn"]

import numpy as np
import h5py

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


def f8_to_c16_interleaved(data):
    """Converts an array of float64 values to complex128 values interleaved."""
    real = data[..., ::2]
    imag = data[..., 1::2]
    return real + 1j * imag


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

    def __init__(self, gkspc: GkSpace, k_weight: float, numbnd: int, is_noncolin: bool):
        if not isinstance(gkspc, GkSpace):
            raise TypeError(
                f"'gkspc' must be a {GkSpace} instance. " f"got {type(gkspc)}."
            )

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
            raise TypeError(
                f"'numbnd' must be a positive integer. "
                f"got {numbnd} (type {type(numbnd)})."
            )
        self.numbnd: int = numbnd
        """Number of KS bands stored
        """

        if not isinstance(is_noncolin, bool):
            raise TypeError(
                f"'is_noncolin' must be a boolean. " f"got '{type(is_noncolin)}'."
            )
        self.is_noncolin: bool = is_noncolin
        """If True, wavefunctions are spin-dependent.
        For non-colinear calculations
        """

        WavefunG = get_WavefunG(self.gkspc, 1 + self.is_noncolin)
        self.evc_gk: WavefunGType = WavefunG.empty(self.numbnd)
        """Contains the KS eigen-wavefunctions
        """
        self.evl: NDArray = np.empty(self.numbnd, dtype="f8", like=self.evc_gk.data)
        """Eigenvalues of the eigenkets in `evc_gk`
        """
        self.occ: NDArray = np.empty(self.numbnd, dtype="f8", like=self.evc_gk.data)
        """Occupation number of the eigenkets in `evc_gk`
        """

    # @profile(precision=4)
    def init_random(self):
        """Initializes `evc_gk` with an unnormalized randomized
        wavefunction, optimized for reduced memory consumption."""

        rng_mod = get_rng_module(self.evc_gk.data)
        seed_k = np.array(self.k_cryst).view("uint")
        rng = rng_mod.default_rng([seed_k, qtmconfig.rng_seed])
        data = self.evc_gk.data

        # Generate random values in chunks to reduce memory usage
        chunk_size = max(1, 10)
        #          = max(1, data.shape[0] // int(data.shape[0]/10))  # Adjust chunk size as needed
        for i in range(0, data.shape[0], chunk_size):
            chunk = slice(i, i + chunk_size)
            # rng.random(data[chunk].shape, out=data[chunk], dtype=data.dtype)
            # data[chunk] += 1j*rng.random(data[chunk].shape)
            random_real = rng.random(data[chunk].shape)
            random_imag = rng.random(data[chunk].shape)

            np.multiply(random_real, np.exp(TPIJ * random_imag), out=data[chunk])

            # In-place multiplication
            # data[chunk] = random_real[:]
            # data[chunk] += 1j * random_imag
            # del random_real, random_imag  # Free memory immediately

        # Old code, before memory optimization using chunks
        # np.multiply(
        #     rng.random(data.shape), np.exp(TPIJ * rng.random(data.shape)), out=data
        # )

        self.evc_gk /= 1 + self.gkspc.gk_norm2

    def compute_rho(
        self, ibnd: slice | Sequence[int] = slice(None), ret_raw=False, normalize=False
    ) -> FieldRType:
        """Constructs a density from the eigenkets `evc_gk` and occupation
        `occ`"""
        self.evc_gk[ibnd].normalize()
        rho: FieldRType = sum(
            occ * wfn.to_r().get_density(normalize=normalize)
            for wfn, occ in zip(self.evc_gk[ibnd], self.occ[ibnd])
        )
        rho /= rho.gspc.reallat_cellvol
        if ret_raw:  # Return both spin components
            return rho
        return rho if self.is_noncolin else rho[0]

    def overlap(
        bra,
        ket: KSWfn,
        bra_bands: Union[int, List[int]],
        ket_bands: Union[int, List[int]],
        umklapp_vec: List[int] = [0, 0, 0],
    ):
        """Calculate psi_self^*(r) . psi_other(r).
        So `reduce=True` will give inner products.
        If the results are not to be cached, pass self_cache or other_cache=False, as required.
        """
        # bra_evc = bra.evc_gk[0, bra_bands, :]
        # ket_evc = ket.evc_gk[0, ket_bands, :]

        def idxgrid_to_dict(
            meanfieldwfn: KSWfn, umkl: List[int], bands: Union[int, List[int]]
        ):
            hashed_indices = bra.gcryst2int(meanfieldwfn.gkspc, umkl)
            dictionary = {}
            for i in range(meanfieldwfn.gkspc.size_g):
                dictionary[hashed_indices[i]] = meanfieldwfn.evc_gk.data[bands, i]
            return dictionary

        bra_dict = idxgrid_to_dict(bra, [0, 0, 0], bra_bands)
        ket_dict = idxgrid_to_dict(ket, umklapp_vec, ket_bands)

        dot = np.zeros((len(bra_bands), len(ket_bands)), dtype=complex)
        for key in bra_dict:
            if key in ket_dict:
                dot += np.outer(np.conjugate(bra_dict[key]), ket_dict[key])

        return dot

    @property
    def indices_occupied(self):
        # print(np.where(self.occ >= 0.5))
        return tuple(np.where(self.occ >= 0.5)[0])

    @property
    def indices_empty(self):
        return tuple(np.where(self.occ < 0.5)[0])

    def gcryst2int(self, gkspc: GkSpace, umklapp: List[int] = [0, 0, 0]):
        cryst = gkspc.g_cryst
        cryst = cryst + np.array(umklapp)[:, None]
        cryst = np.mod(cryst.T, gkspc.grid_shape).T
        hash = (
            cryst[0, :]
            + cryst[1, :] * gkspc.grid_shape[0]
            + cryst[2, :] * gkspc.grid_shape[0] * gkspc.grid_shape[1]
        )
        return hash

    def init_from_hdf5(self, h5file: str):
        """Initializes the wavefunctions from an HDF5 file written in Quantum ESPRESSO format."""

        with h5py.File(h5file, "r") as f:
            # Read MillerIndices
            miller_indices = np.array(f["MillerIndices"][:])
            # Read wavefunctions
            evc = np.array(f["evc"][:])
            if "evl" in f.attrs:
                evl = np.array(f.attrs["evl"][:])
            else:
                evl = None

        if evl is not None:
            self.evl[:] = evl[0, :]

        # Convert wavefunctions to complex format
        evc = f8_to_c16_interleaved(evc) * np.prod(self.gkspc.grid_shape)

        if self.gkspc.size_g != len(miller_indices):
            raise ValueError(
                f"Number of Miller indices in HDF5 file ({len(miller_indices)}) does not match the size of gkspc.g_cryst ({self.gkspc.size_g})"
            )

        for i in range(len(miller_indices)):
            idx = np.where(np.all(self.gkspc.g_cryst.T == miller_indices[i], axis=1))
            if len(idx) > 0:
                self.evc_gk.data[:, idx[0][0]] = evc[:, i]
            else:
                raise ValueError(
                    "Miller index {i}:{miller_indices[i]} not found in gkspc.g_cryst"
                )

        self.evc_gk.normalize()
