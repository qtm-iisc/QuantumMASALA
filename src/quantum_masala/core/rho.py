# TODO: Check Documentation
from typing import Optional
from warnings import warn

import numpy as np

from .cryst import Crystal
from .gspc import GSpace
from .fft import FFTGSpace
from .symm import SymmGspc
from .mpicomm import PWComm

EPS5 = 1E-5
EPS10 = 1E-10


class Rho:
    r"""Represents (spin-dependent) electron density in crystal

    Contains both core and valence electron data. Methods to update density are implemented,
    which involves normalization and ensuring that it exhibits the same symmetry as the crystal.

    Attributes
    ----------
    grho : GSpace
        Represents the G-space truncated based on Kinetic Energy Cutoff for hard-grid.
    fft_rho : FFTGSpace
        FFT Interface for `grho`; Required for representing density as a function of 'r' or 'g'.
    numspin : {1, 2}
        1 if calculation is non-polarized, 2 if spin-polarized (LSDA).
    numel : float
        Total number of electrons per unit cell.
    symm_flag : bool
        If `True`, symmetrizes the charge density `rho` such that it has the same symmetry as
        the crystal

    core_r, core_g : np.ndarray
        Arrays representing (Spin-independent) density of core electrons across real-space (`r`)
        and reciprocal-space (`g`) respectively.
    r, g : np.ndarray
        Arrays representing (Spin-depedent) density of core electrons across real-space (`r`)
        and reciprocal-space (`g`) respectively.

    Methods
    -------
    sync():
        Updates charge density across all processes to be the same.
    integral_rho_f_dv(f_r: np.ndarray, sum_over_spin: bool = True):
        Computes :math:`\int_{\Omega} \dd{r} \rho_i(\mathbf{r}) f(\mathbf{r})` for input array `f_r`
        if `sum_over_spin` is True, the values across spins are summed.
    update_rho(rho_new: np.ndarray):
        Updates the charge density from input array. The type of representation is determined from
        the input array's shape. Values are normalized and 'symmetrized' to ensure that the density
        has the same symmetry as the crystal.

    """

    pwcomm: PWComm = PWComm()
    grho: GSpace
    fft_rho: FFTGSpace
    numspin: int
    numel: float
    symm_flag: bool
    core_r: np.ndarray
    core_g: np.ndarray

    def __init__(
        self,
        crystal: Crystal,
        grho: GSpace,
        numspin: int,
        fft_rho: Optional[FFTGSpace] = None,
        rho: Optional[np.ndarray] = None,
        rhocore: Optional[np.ndarray] = None,
        symm_flag: bool = True
    ):
        self.grho = grho

        if fft_rho is None:
            fft_rho = FFTGSpace(self.grho)
        self.fft_rho = fft_rho

        if numspin not in [1, 2]:
            raise ValueError(f"'numspin' must be either '1' or '2'. Got {numspin}")
        self.numspin = numspin

        self.numel = crystal.numel

        self.symm_flag = symm_flag
        if symm_flag:
            self._symmmod = SymmGspc(crystal, self.grho)
        else:
            self._symmmod = None

        self.core_g = np.empty(self.grho.numg, dtype="c16")
        self.core_r = np.empty(self.grho.grid_shape, dtype="c16")
        if rhocore is not None:
            if rhocore.shape == (self.grho.numg,):
                self.core_g[:] = rhocore
                self.core_r = self.fft_rho.g2r(self.core_g, self.core_r)
            elif rhocore.shape == self.grho.grid_shape:
                self.core_r[:] = rhocore
                self.core_g = self.fft_rho.r2g(self.core_r, self.core_g)
            else:
                raise ValueError(
                    f"'rhocore.shape' must be either {(self.grho.numg,)} or {self.grho.grid_shape}. "
                    f"Got {rhocore.shape}"
                )
        else:
            self.core_g[:] = 0
            self.core_r[:] = 0

        self._g = np.empty((self.numspin, self.grho.numg), dtype="c16")
        self._r = np.empty((self.numspin, *self.grho.grid_shape), dtype="c16")

        if rho is not None:
            self.update(rho)

    @property
    def r(self) -> np.ndarray:
        return self._r

    @property
    def g(self) -> np.ndarray:
        return self._g

    @property
    def aux_r(self) -> np.ndarray:
        r"""Spin-dependent core+valence charge density in real-space :math:`\rho(\mathbf{r})`"""
        return self._r + self.core_r

    @property
    def aux_g(self) -> np.ndarray:
        r"""Spin-dependent core+valence charge density in reciprocal-space :math:`\rho(\mathbf{G})`"""
        return self._g + self.core_g

    def sync(self) -> None:
        self._g = self.pwcomm.world_comm.Bcast(self._g)
        self._r = self.fft_rho.g2r(self._g)

    def integral_rho_f_dv(
        self, f_r: np.ndarray, sum_over_spin: bool = True
    ) -> np.ndarray:
        grid_shape = self.grho.grid_shape
        if f_r.shape[-3:] != grid_shape:
            try:
                np.broadcast_to(f_r, grid_shape)
            except ValueError:
                raise ValueError(f"'f_r.shape' must be {grid_shape}. Got {f_r.shape}")

        if sum_over_spin:
            return self.grho.reallat_dv * np.sum(f_r * self.r, axis=(-1, -2, -3, -4))
        else:
            return self.grho.reallat_dv * np.sum(f_r * self.r, axis=(-1, -2, -3))

    def _normalize(self) -> None:
        rho_int = np.sum(self._g[:, 0]) * self.grho.reallat_dv
        rho_int *= (1 + (self.numspin == 1))

        if rho_int < EPS10:
            raise ValueError("values in 'rho.r' too small to normalize.\n"
                             f"computed total charge = {rho_int}")
        if np.abs(rho_int - self.numel) > EPS5:
            warn(f"total charge renormalized from {rho_int} to {self.numel}")
        fac = self.numel / rho_int
        self._g *= fac
        self._r *= fac

    def update(self, rho_new: np.ndarray) -> None:
        if rho_new.shape[0] != self.numspin:
            raise ValueError(
                f"'rho_new.shape[0]' must be equal to 'numspin'. "
                f"Expected {self.numspin}, got {rho_new.shape[0]}"
            )
        if rho_new.shape[1:] == (self.grho.numg,):
            self._r[:] = self.fft_rho.g2r(rho_new)
        elif rho_new.shape[1:] == self.grho.grid_shape:
            self._r[:] = rho_new
        else:
            raise ValueError(
                f"'rho_new.shape[1:]' must be either {(self.grho.numg,)} (in G-space) "
                f"or {self.grho.grid_shape} (in Real Space). Got {rho_new.shape[1:]}"
            )

        rho_neg = np.abs(self._r) - self._r.real
        i_rho_neg = np.nonzero(rho_neg > EPS5)
        if len(i_rho_neg[0]) != 0:
            del_rho = np.sum(np.abs(self._r[i_rho_neg])) * self.grho.reallat_dv
            warn("negative/complex values found in `rho.r`.\n"
                 f"Error: {del_rho}")
        self._r[:] = np.abs(self._r[:], out=self._r[:])

        self._g[:] = self.fft_rho.r2g(self._r, self._g)
        if self.symm_flag:
            self._g[:] = self._symmmod.symmetrize(self._g)

        self.sync()
        self._normalize()
