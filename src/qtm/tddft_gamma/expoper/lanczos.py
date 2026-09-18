r"""Lanczos / Krylov-subspace propagator: rather than approximating
:math:`\exp(-i\,dt\,H)` uniformly over a declared global spectral range
(`qtm.tddft_gamma.expoper.kte.KTEExp`) or as a power series around a fixed
reference point (`qtm.tddft_gamma.expoper.taylor.TaylorExp`), this builds
an `m`-dimensional Krylov subspace `{psi, H*psi, H^2*psi, ...}` FROM THE
ACTUAL STATE BEING PROPAGATED, tridiagonalizes `H` on it
(`qtm.linalg.lanczos.lanczos_tridiagonalize`), and applies the exponential
of that small `m x m` matrix exactly:

.. math::
    \exp(-i\,dt\,H)\,\psi \;\approx\; \|\psi\|\; Q_m\, \exp(-i\,dt\,T_m)\,
    e_1

`T_m` is real symmetric tridiagonal (since `H` is Hermitian), so its
exponential is obtained via `scipy.linalg.eigh_tridiagonal` rather than a
generic (dense, complex) matrix exponential -- diagonalize once,
`\exp(-i\,dt\,T_m) e_1 = V\,\mathrm{diag}(e^{-i\,dt\,\lambda_k})\,V^T e_1`,
same approach as the "short iterative Lanczos" (SIL) method (Park &
Light, 1986).

The point of doing this instead of KTE: convergence here is governed by
the spread of the Ritz values -- i.e. by whatever part of `H`'s spectrum
`psi` actually has weight on -- not by the full declared `E_max - E_min`.
For a state that is spectrally localized well inside a much wider
Hamiltonian spectrum (e.g. a smooth valence-band state in a plane-wave
basis whose kinetic-energy cutoff sets a far larger `E_max` than the
state ever populates), this can converge in substantially fewer matvecs
than KTE, which has no way to know `psi` never reaches those energies.

Two real costs come with that adaptivity, both absent from `KTEExp`:

* No a priori order formula. `KTEExp`'s Chebyshev order is computed in
  advance from `z = dE*dt` alone, before touching `psi`. Here, `m` is
  found by monitoring successive approximations as `m` grows and
  stopping once they stop changing (`tol`), which means the cost is only
  known after the fact and can vary between bands/steps -- and, for a
  genuinely time-dependent `H` (as in a real, self-consistent TDDFT run),
  can drift upward over the course of a simulation if the dynamics
  broadens `psi`'s spectral content, which `KTEExp`'s fixed-cost-per-step
  guarantee does not share.
* Reorthogonalization. Lanczos vectors lose mutual orthogonality to
  rounding error after enough iterations; `lanczos_tridiagonalize` always
  reorthogonalizes fully against every previous vector, costing `O(N*m)`
  per step on top of the matvec -- a cost `KTEExp`'s three-term recursion
  (no basis to keep orthogonal) simply doesn't have. How much this
  matters in practice depends entirely on the relative cost of a single
  matvec: against an `O(N log N)`-FFT-dominated `h_psi`, `O(N*m)`
  reorthogonalization at the small `m` a well-localized state needs is
  comparatively cheap; against a `matvec` that is itself just an `O(N)`
  operation (no FFT, e.g. a purely diagonal or otherwise sparse `H`),
  reorthogonalization would matter far more relative to it.
"""
__all__ = ["LanczosExp"]

import numpy as np
from scipy.linalg import eigh_tridiagonal

from qtm.containers.field import FieldRType
from qtm.containers.wavefun import get_WavefunG
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.logger import qtmlogger
from qtm.pseudo.nloc import NonlocGenerator

from qtm.linalg.bicgstab import _global_vdot
from qtm.linalg.lanczos import lanczos_steps

from .base import TDExpOperBase


class LanczosExp(TDExpOperBase):
    __slots__ = ["tol", "m_max"]

    def __init__(
        self,
        gkspc: GkSpace,
        is_spin: int,
        is_noncolin: bool,
        vloc: FieldRType,
        l_nloc: list[NonlocGenerator],
        time_step: float,
        tol: float = 1e-10,
        m_max: int = 60,
    ):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc, time_step)
        self.tol = tol
        """Stopping tolerance: iterate until successive Krylov-subspace
        approximations change by less than this (relative norm)."""
        self.m_max = m_max
        """Hard cap on Krylov subspace size, in case `tol` is not reached
        (a warning is logged if so, per band)."""

    def _h_linear(self, x: np.ndarray) -> np.ndarray:
        """Applies the full Hamiltonian to a single-band G-space
        coefficient vector `x` (same pattern as
        `CrankNicolson._h_linear`)."""
        WavefunG = get_WavefunG(self.gkspc, 1)
        psi = WavefunG.empty(())
        psi.data[:] = x
        hpsi = WavefunG.empty(())
        self.h_psi(psi, hpsi)
        return hpsi.data

    def _propagate_one(self, psi0: np.ndarray) -> np.ndarray:
        beta0 = np.sqrt(_global_vdot(self.gkspc, psi0, psi0).real)
        prev_approx = None

        for m_eff, (Q, alphas, betas) in enumerate(
            lanczos_steps(self._h_linear, self.gkspc, psi0), start=1
        ):
            if m_eff == 1:
                ritz_vals, ritz_vecs = alphas, np.array([[1.0]])
            else:
                ritz_vals, ritz_vecs = eigh_tridiagonal(alphas, betas)
            phase = np.exp(-1j * self.time_step * ritz_vals)
            small = ritz_vecs @ (phase * ritz_vecs[0, :])
            approx = beta0 * sum(c * qc for c, qc in zip(small, Q))

            if prev_approx is not None:
                num = np.sqrt(
                    _global_vdot(
                        self.gkspc, approx - prev_approx, approx - prev_approx
                    ).real
                )
                den = np.sqrt(_global_vdot(self.gkspc, approx, approx).real)
                if den > 0 and num / den < self.tol:
                    return approx
            prev_approx = approx

            if m_eff >= self.m_max:
                qtmlogger.warning(
                    f"LanczosExp: did not reach tol={self.tol} within "
                    f"m_max={self.m_max} Krylov iterations; returning the "
                    "last iterate."
                )
                return approx

        # `lanczos_steps` stopped on its own: an invariant subspace was
        # found exactly, which is a genuine convergence, not a failure.
        return prev_approx

    def prop_psi(self, l_psi_in: list[KSWfn], l_psi_out: list[KSWfn]):
        """Propagates `l_psi_in` by one full time step via the Lanczos
        (Krylov-subspace) expansion, writing the result into `l_psi_out`
        (which may be the same object(s) as `l_psi_in`, or different ones
        -- either way, `l_psi_in` is left untouched)."""
        if self.is_noncolin:
            qtmlogger.warning("LanczosExp.prop_psi(): is_noncolin not implemented yet.")
            return

        size_g = self.gkspc.size_g

        for idxspin in range(1 + self.is_spin * (not self.is_noncolin)):
            if self.is_spin:
                self.set_idxspin(idxspin)

            data_in = l_psi_in[idxspin].evc_gk.data.reshape(-1, size_g).copy()
            l_psi_out[idxspin].evc_gk[:] = l_psi_in[idxspin].evc_gk[:]
            data_out = l_psi_out[idxspin].evc_gk.data.reshape(-1, size_g)

            for iband in range(data_in.shape[0]):
                data_out[iband] = self._propagate_one(data_in[iband])
