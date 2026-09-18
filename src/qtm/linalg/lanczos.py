r"""Lanczos tridiagonalization of a Hermitian linear operator, and
spectral-bound estimation built on top of it.

Every inner product goes through `_global_vdot` (see `qtm.linalg.bicgstab`),
so both functions here are safe under an MPI-distributed `GkSpace` the same
way `bicgstab_dist` is.

`lanczos_bounds` estimates `(e_min, e_max)` from whatever probe vector the
caller supplies -- it makes no assumption about what "generic" means for
that vector, since that depends on the caller's own situation (whether it
wants a state-independent bound, or specifically wants to exploit a
particular vector's spectral localization). `margin` pads the result to
cover the gap between "this probe's Lanczos iteration converged" and "this
is truly the operator's global extremal eigenvalue": a probe with
negligible overlap on whichever eigenvector sets the true extremum will
still underestimate the range no matter how many iterations are run
against it, and a caller relying on the padded range to stay within `[-1,
1]` after rescaling (as a Chebyshev-based expansion would) should treat an
underestimate as a correctness risk, not just a slower-converging one.
"""
from __future__ import annotations

__all__ = ["lanczos_steps", "lanczos_tridiagonalize", "lanczos_bounds"]

import itertools
from typing import Callable, Optional

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .bicgstab import _global_vdot


def _extremal_ritz(alphas: np.ndarray, betas: np.ndarray):
    """Diagonalizes the (small, real, tridiagonal) `m x m` matrix built so
    far and returns its full set of Ritz values/vectors. `m` stays small
    enough (tens, not thousands) that this costs nothing next to the
    matvecs that built it -- see `lanczos_bounds`'s docstring for why that
    matters."""
    if alphas.size == 1:
        return alphas, np.ones((1, 1))
    return eigh_tridiagonal(alphas, betas)


def lanczos_steps(
    matvec: Callable[[np.ndarray], np.ndarray],
    gkspc,
    psi0: np.ndarray,
    reorth: bool = True,
):
    """Generator form of the Lanczos three-term recurrence: adds one
    basis vector to the Krylov subspace per iteration and yields the
    running `(Q, alphas, betas)` after each addition (same meaning as
    `lanczos_tridiagonalize`'s return value, but growing one vector at a
    time), stopping on its own once an invariant subspace is found
    exactly.

    This is the shared engine behind both `lanczos_tridiagonalize` (which
    just takes a fixed number of steps from it) and any caller that needs
    to inspect the subspace after every new vector to decide adaptively
    when to stop, such as a Krylov-subspace propagator checking whether
    successive approximations to `exp(-i dt H) psi` have converged.

    Uses full reorthogonalization against every previous basis vector at
    each step: Lanczos vectors lose mutual orthogonality to rounding
    error after a handful of iterations, and the number of iterations
    actually taken is expected to stay small enough (tens, not thousands)
    that the resulting O(m^2) cost is negligible next to the O(m)
    matvecs.
    """
    beta0 = np.sqrt(_global_vdot(gkspc, psi0, psi0).real)
    q = psi0 / beta0
    Q: list[np.ndarray] = []
    alphas: list[float] = []
    betas: list[float] = []
    q_prev = np.zeros_like(psi0)
    beta = 0.0

    while True:
        j = len(Q)
        Q.append(q)
        w = matvec(q)
        alpha = _global_vdot(gkspc, q, w).real
        alphas.append(alpha)
        w = w - alpha * q - (beta * q_prev if j > 0 else 0.0)
        if reorth:
            for qc in Q:
                w = w - _global_vdot(gkspc, qc, w) * qc
        beta = np.sqrt(_global_vdot(gkspc, w, w).real)
        q_prev = q
        yield Q, np.array(alphas), np.array(betas)
        if beta < 1e-13 * max(beta0, 1.0):
            return
        betas.append(beta)
        q = w / beta


def lanczos_tridiagonalize(
    matvec: Callable[[np.ndarray], np.ndarray],
    gkspc,
    psi0: np.ndarray,
    m: int,
    reorth: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Builds an up-to-`m`-dimensional Krylov basis for `matvec` (assumed
    Hermitian) starting from `psi0`, via `lanczos_steps`.

    Returns `(Q, alphas, betas)`: `Q` is the list of `m_eff` orthonormal
    basis vectors (`m_eff <= m`, shorter if an invariant subspace is found
    exactly first), `alphas` (length `m_eff`) and `betas` (length
    `m_eff - 1`) are the real diagonal/off-diagonal entries of the
    resulting tridiagonal matrix `T_m = Q^dagger H Q`.
    """
    result = ([], np.array([]), np.array([]))
    for result in itertools.islice(lanczos_steps(matvec, gkspc, psi0, reorth), m):
        pass
    return result


def lanczos_bounds(
    matvec: Callable[[np.ndarray], np.ndarray],
    gkspc,
    psi0: np.ndarray,
    n_iter: int = 30,
    margin: float = 0.005,
    return_vectors: bool = False,
    check_every: int = 5,
    conv_tol: Optional[float] = None,
):
    """Estimates `(e_min, e_max)` for `matvec` by Lanczos-tridiagonalizing
    it (see `lanczos_steps`) starting from `psi0`, then padding the
    resulting extremal Ritz values by `margin` (a fraction of the
    estimated range, on each side).

    `n_iter` is a CAP, not a fixed count: the extremal Ritz values of a
    well-separated spectrum typically stop moving well before `n_iter`
    Krylov vectors have been added, and diagonalizing the (small,
    real-tridiagonal) matrix built so far costs nothing next to another
    matvec -- so every `check_every` vectors, the current extremal Ritz
    values are compared to the previous checkpoint's, and iteration stops
    as soon as their change (relative to the current estimated range) is
    below `conv_tol`. `conv_tol` defaults to `margin` itself: there is no
    point refining the raw estimate any tighter than the fraction it is
    about to be padded by anyway, since that padding already absorbs
    exactly this much residual error.

    If `return_vectors` is set, also returns the (full-space, normalized)
    Ritz vectors `(v_min, v_max)` corresponding to the extremal Ritz
    values, as `(e_min, e_max), v_min, v_max` -- useful for warm-starting a
    later re-estimate: if the operator's true extremal eigenvalues have
    only moved a little since this call, `v_min`/`v_max` are already close
    to the new extremal eigenvectors, so a probe seeded from them should
    re-converge in far fewer iterations than a fresh probe would need.
    """
    if conv_tol is None:
        conv_tol = margin

    prev_bounds = None
    state = None
    for m_eff, state in enumerate(lanczos_steps(matvec, gkspc, psi0), start=1):
        _, alphas, betas = state
        if m_eff % check_every != 0 and m_eff < n_iter:
            continue
        ritz_vals, _ = _extremal_ritz(alphas, betas)
        e_min, e_max = float(ritz_vals.min()), float(ritz_vals.max())
        if prev_bounds is not None:
            scale = max(e_max - e_min, 1e-12)
            changed = max(abs(e_min - prev_bounds[0]), abs(e_max - prev_bounds[1]))
            if changed / scale < conv_tol:
                break
        prev_bounds = (e_min, e_max)
        if m_eff >= n_iter:
            break

    Q, alphas, betas = state
    ritz_vals, ritz_vecs = _extremal_ritz(alphas, betas)
    idx_min, idx_max = int(np.argmin(ritz_vals)), int(np.argmax(ritz_vals))
    e_min, e_max = float(ritz_vals[idx_min]), float(ritz_vals[idx_max])
    pad = margin * max(e_max - e_min, 1e-12)
    bounds = (e_min - pad, e_max + pad)
    if not return_vectors:
        return bounds

    Qarr = np.array(Q)
    v_min = Qarr.T @ ritz_vecs[:, idx_min].astype(Qarr.dtype)
    v_max = Qarr.T @ ritz_vecs[:, idx_max].astype(Qarr.dtype)
    v_min = v_min / np.sqrt(_global_vdot(gkspc, v_min, v_min).real)
    v_max = v_max / np.sqrt(_global_vdot(gkspc, v_max, v_max).real)
    return bounds, v_min, v_max
