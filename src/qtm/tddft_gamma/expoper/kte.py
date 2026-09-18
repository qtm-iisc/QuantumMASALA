r"""Kosloff-Tal-Ezer (KTE) propagator: expands :math:`\exp(-i\,dt\,H)` in
Chebyshev polynomials of a rescaled Hamiltonian, rather than in a Taylor
series of `H` directly (`qtm.tddft_gamma.expoper.taylor.TaylorExp`).

Writing :math:`H = \bar E\, I + \Delta E\, \tilde H`, with :math:`\bar E =
(E_{\max}+E_{\min})/2` and :math:`\Delta E = (E_{\max}-E_{\min})/2` chosen
so :math:`\tilde H`'s spectrum lies in :math:`[-1, 1]`:

.. math::
    \exp(-i\,dt\,H) = e^{-i\,dt\,\bar E}\; \exp(-i z \tilde H),
    \qquad z = \Delta E \cdot dt

and expanding the remaining factor in Chebyshev polynomials
:math:`T_n(\tilde H)` gives, via the Jacobi-Anger identity, coefficients
that are ordinary Bessel functions: :math:`a_0 = J_0(z)`, :math:`a_n =
2(-i)^n J_n(z)` for :math:`n \geq 1`. `T_n(\tilde H)\psi` is built by the
same three-term recurrence as the polynomials themselves,
:math:`T_{n+1}(\tilde H)\psi = 2\tilde H\, T_n(\tilde H)\psi -
T_{n-1}(\tilde H)\psi`, so each term costs exactly one application of
`H` (`h_psi`/`_h_linear`), same as one iteration of `TaylorExp`.

Why bother when `TaylorExp` already converges: for a FIXED order, this is
the (near-)minimax polynomial approximation to :math:`e^{-iz x}` on
`[-1,1]`, while a Taylor series is only optimal at a single point (`x=0`)
-- so for the same target accuracy, this needs noticeably fewer terms
(and stays numerically well-conditioned doing it: `|a_n| <= 1` for every
`n`, whereas `TaylorExp`'s un-shifted terms can transiently grow far
larger than the final result before decaying, costing precision to
cancellation).

`(E_min, E_max)` are estimated exactly ONCE per `KTEExp` instance -- on
its first `prop_psi` call, from a random probe vector (`_random_probe`,
not the physical wavefunction being propagated -- see its docstring for
why), via `qtm.linalg.lanczos.lanczos_bounds` -- and then cached in
`self._e_bounds` for the lifetime of the instance, deliberately NOT
refreshed on later calls even as `vloc` is updated between TDDFT steps.
Bound estimation is itself a Lanczos run whose cost is comparable to the
propagation step it enables, so recomputing it on every call can cost
more than the lower Chebyshev order it buys. It also isn't necessary:
`H`'s spectral RANGE drifts far more slowly across a self-consistent
TDDFT run than `psi` itself does, so a single conservatively-padded
estimate stays valid for the whole run rather than needing to track that
drift exactly.

For a run long/aggressive enough that this stops holding, `refresh_every`
re-estimates the bounds periodically rather than never -- but seeded from
the PREVIOUS extremal Ritz vectors (`_v_min`/`_v_max`), not a fresh random
probe: since the true range is assumed to have drifted only a little
since they were found (the same assumption `margin` relies on), each is
already close to the corresponding new extremal eigenvector, so a probe
started there should re-converge in far fewer iterations than starting
over from scratch would need. See `_refresh_bounds`.

This is the piece of KTE that is genuinely more delicate than
`TaylorExp`: an underestimated `E_max - E_min` doesn't just converge
slower here, it can make the recursion numerically unstable (`T_n(x)`
grows without bound for `|x| > 1`).

Order selection is the other half of KTE's usual appeal -- unlike
`TaylorExp` (which takes a fixed `order` and offers no guidance on
picking it), the number of terms needed for a target tolerance follows
directly from `z` via how fast `|a_n|` decays past the Bessel-function
turning point `n ~ z`, computed here by direct evaluation
(`scipy.special.jv`) rather than the large-`z` asymptotic formula, which
is not reliable at the small-to-moderate `z` typical of a single TDDFT
time step.
"""
__all__ = ["KTEExp"]

from typing import Optional

import numpy as np
from scipy.special import jv

from qtm.containers.field import FieldRType
from qtm.containers.wavefun import get_WavefunG
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.logger import qtmlogger
from qtm.pseudo.nloc import NonlocGenerator

from qtm.linalg.bicgstab import _global_vdot
from qtm.linalg.lanczos import lanczos_bounds

from .base import TDExpOperBase


def _kte_coeffs(z: float, tol: float, n_cap: int) -> np.ndarray:
    """Returns the Chebyshev coefficients `a_n = 2*(-i)^n*J_n(z)` (`a_0`
    not doubled), truncated at the smallest `n` where two consecutive
    coefficients both fall below `tol` in magnitude (a cheap safeguard
    against stopping in a transient dip before the asymptotic decay past
    the Bessel turning point `n ~ z` has actually set in)."""
    n_max = max(n_cap, int(z) + 50)
    ns = np.arange(n_max + 1)
    bessel = jv(ns, z)
    below = np.abs(bessel) < tol
    both_below = below[:-1] & below[1:]
    hit = np.flatnonzero(both_below)
    order = int(hit[0]) if hit.size else n_max
    coeffs = 2 * (-1j) ** ns[: order + 1] * bessel[: order + 1]
    coeffs[0] = bessel[0]
    return coeffs


class KTEExp(TDExpOperBase):
    __slots__ = [
        "tol",
        "margin",
        "n_probe",
        "n_probe_refresh",
        "n_cap",
        "refresh_every",
        "_e_bounds",
        "_v_min",
        "_v_max",
        "_step_count",
    ]

    def __init__(
        self,
        gkspc: GkSpace,
        is_spin: int,
        is_noncolin: bool,
        vloc: FieldRType,
        l_nloc: list[NonlocGenerator],
        time_step: float,
        tol: float = 1e-10,
        margin: float = 0.005,
        n_probe: int = 30,
        n_probe_refresh: int = 5,
        n_cap: int = 200,
        refresh_every: Optional[int] = None,
    ):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc, time_step)
        self.tol = tol
        """Target relative error for the Chebyshev truncation."""
        self.margin = margin
        """Fractional safety padding applied to the estimated spectral
        range -- see `qtm.linalg.lanczos.lanczos_bounds`'s docstring."""
        self.n_probe = n_probe
        """Lanczos iterations for the INITIAL (cold-start, random-probe)
        bound estimate, which has nothing to seed from and so needs
        enough iterations to find the extremal eigenvectors from
        scratch."""
        self.n_probe_refresh = n_probe_refresh
        """Lanczos iterations for each periodic refresh (see
        `refresh_every`), seeded from the previous extremal Ritz vectors
        rather than a random vector. Deliberately much smaller than
        `n_probe`: physically, `E_max` is set almost entirely by the
        kinetic-energy cutoff (fixed for the whole run, not something
        `vloc` can move) and `E_min` by whichever state is most localized
        under the local potential -- neither moves far in one refresh
        interval unless `vloc` itself has changed by a lot, so the seed
        is already close and only needs a few iterations of correction,
        not a from-scratch search."""
        self.n_cap = n_cap
        """Hard cap on the Chebyshev order, guarding against runaway
        growth if `tol` and the estimated `z` are badly mismatched."""
        self.refresh_every = refresh_every
        """Re-estimate spectral bounds every this many `prop_psi` calls,
        seeded from the previous extremal Ritz vectors rather than a
        fresh random probe (see `_refresh_bounds`). `None` (default)
        means never refresh after the first estimate: `margin` already
        covers ordinary drift, per the class docstring's cost accounting,
        so this exists for runs long/aggressive enough that assumption
        stops holding, not as something every use needs to set."""
        self._e_bounds = None
        """Cached `(E_min, E_max)`. See `refresh_every` and class
        docstring for when/why this is (or isn't) recomputed."""
        self._v_min = None
        self._v_max = None
        """Extremal Ritz vectors from the last bound estimation, kept
        specifically to warm-start the next one (if `refresh_every` is
        set): if the true spectral range has only drifted a little, these
        are already close to the new extremal eigenvectors, so seeding
        from them re-converges far faster than a fresh random probe."""
        self._step_count = 0

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

    def _random_probe(self, dtype) -> np.ndarray:
        """A generic probe vector for spectral-bound estimation --
        independently random per MPI rank's local slice of G-space.

        This does NOT need to be reproducible from one global seed across
        ranks (a genuinely harder problem, needing a global-to-local
        G-vector index map to slice consistently): a Lanczos probe only
        needs generic overlap across the spectrum -- i.e. not to be
        orthogonal to whichever eigenvector sets the true extremum -- and
        every rank independently drawing i.i.d. random values for its own
        local coefficients already produces a bona fide random vector
        over the FULL distributed Hilbert space. No cross-rank
        coordination beyond the ordinary Allreduce-based inner products
        `lanczos_tridiagonalize` already performs is needed.

        Used in place of the wavefunction actually being propagated
        specifically so the estimate does not inherit whatever spectral
        localization that physical state happens to have (the opposite
        situation from `LanczosExp`, which WANTS that localization) --
        see `qtm.linalg.lanczos` for the general caveat this narrows but
        does not eliminate: a random vector still
        converges more slowly onto an isolated, weakly-coupled extremal
        eigenvector than a generous `n_probe` can guarantee in the
        general case.
        """
        rng = np.random.default_rng()
        size_g = self.gkspc.size_g
        vec = rng.standard_normal(size_g) + 1j * rng.standard_normal(size_g)
        return vec.astype(dtype)

    def _refresh_bounds(self, dtype) -> None:
        """(Re-)estimates `self._e_bounds`, and updates `self._v_min`,
        `self._v_max` for next time.

        On the very first call (`self._v_min is None`), probes from a
        fresh random vector for `self.n_probe` iterations (`_random_probe`
        -- there is nothing to warm-start from yet, so this needs enough
        iterations to find the extremal eigenvectors from scratch). On
        every later call (only reached at all if `refresh_every` is set),
        instead runs TWO much shorter probes (`self.n_probe_refresh`),
        one seeded from each of the previous extremal Ritz vectors: `E_max`
        is set almost entirely by the (fixed) kinetic-energy cutoff and
        `E_min` by whichever state is most localized under the local
        potential, so ordinary `vloc` drift between refreshes moves
        either only a little, and a probe started from the previous
        answer only needs a few iterations of correction, not a
        from-scratch search. Both probes' Ritz values are pooled
        (`e_min` = the smaller of the two probes' minima, `e_max` = the
        larger of their maxima) since either seed could in principle also
        pick up the other extremum.
        """
        if self._v_min is None:
            probe = self._random_probe(dtype)
            bounds, self._v_min, self._v_max = lanczos_bounds(
                self._h_linear,
                self.gkspc,
                probe,
                self.n_probe,
                self.margin,
                return_vectors=True,
            )
            self._e_bounds = bounds
            return

        (e_min_lo, e_max_lo), v_min_new, _ = lanczos_bounds(
            self._h_linear,
            self.gkspc,
            self._v_min,
            self.n_probe_refresh,
            self.margin,
            return_vectors=True,
        )
        (e_min_hi, e_max_hi), _, v_max_new = lanczos_bounds(
            self._h_linear,
            self.gkspc,
            self._v_max,
            self.n_probe_refresh,
            self.margin,
            return_vectors=True,
        )
        self._e_bounds = (min(e_min_lo, e_min_hi), max(e_max_lo, e_max_hi))
        self._v_min, self._v_max = v_min_new, v_max_new

    def prop_psi(self, l_psi_in: list[KSWfn], l_psi_out: list[KSWfn]):
        """Propagates `l_psi_in` by one full time step via the Chebyshev
        (KTE) expansion, writing the result into `l_psi_out` (which may be
        the same object(s) as `l_psi_in`, or different ones -- either way,
        `l_psi_in` is left untouched)."""
        if self.is_noncolin:
            qtmlogger.warning("KTEExp.prop_psi(): is_noncolin not implemented yet.")
            return

        size_g = self.gkspc.size_g

        if self._e_bounds is None:
            self._refresh_bounds(l_psi_in[0].evc_gk.data.dtype)
        elif (
            self.refresh_every is not None
            and self._step_count % self.refresh_every == 0
        ):
            self._refresh_bounds(l_psi_in[0].evc_gk.data.dtype)
        self._step_count += 1

        e_min, e_max = self._e_bounds
        e_bar = 0.5 * (e_min + e_max)
        d_e = 0.5 * (e_max - e_min)
        z = d_e * self.time_step
        # Cheap relative to `lanczos_bounds` (a single vectorized `jv`
        # call vs. a full Lanczos run), so recomputed each call rather
        # than also cached.
        coeffs = _kte_coeffs(z, self.tol, self.n_cap)

        for idxspin in range(1 + self.is_spin * (not self.is_noncolin)):
            if self.is_spin:
                self.set_idxspin(idxspin)

            data_in = l_psi_in[idxspin].evc_gk.data.reshape(-1, size_g).copy()
            l_psi_out[idxspin].evc_gk[:] = l_psi_in[idxspin].evc_gk[:]
            data_out = l_psi_out[idxspin].evc_gk.data.reshape(-1, size_g)

            for iband in range(data_in.shape[0]):
                psi0 = data_in[iband]

                def _htilde(x, e_bar=e_bar, d_e=d_e):
                    return (self._h_linear(x) - e_bar * x) / d_e

                phi_prev = psi0.copy()
                result = coeffs[0] * phi_prev
                if coeffs.size > 1:
                    phi_curr = _htilde(psi0)
                    result = result + coeffs[1] * phi_curr
                    for n in range(2, coeffs.size):
                        phi_next = 2 * _htilde(phi_curr) - phi_prev
                        result = result + coeffs[n] * phi_next
                        phi_prev, phi_curr = phi_curr, phi_next

                data_out[iband] = np.exp(-1j * e_bar * self.time_step) * result
