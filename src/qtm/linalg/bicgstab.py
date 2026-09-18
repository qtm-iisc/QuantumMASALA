r"""A BiCGSTAB implementation whose vector inner products are Allreduced
across a `GkSpace`'s plane-wave-group communicator when it is distributed.

`scipy.sparse.linalg.bicgstab` cannot be used directly on a G-space
coefficient vector that is split across MPI ranks (as it is whenever
`gkspc` is a `qtm.mpi.gspace.DistGkSpace`, e.g. under
`DFTCommMod(..., pwgrp_size>1)`): scipy's solver only ever sees, and only
ever computes inner products/norms over, the local NumPy array it is
handed -- it has no way to know that array is just one rank's slice of a
larger distributed vector. Two consequences follow, both serious, and both
confirmed directly on `SplitOper._oper_vloc_cayley`'s and
`CrankNicolson._h_linear`'s use of ``scipy.sparse.linalg.bicgstab``, whose
`matvec` (`_vloc_linear`/`h_psi`) performs a genuine cross-rank collective
(`to_r()`/`to_g()`) that only makes sense applied to the FULL vector:

1. Every scalar coefficient the algorithm computes (rho, alpha, omega, the
   residual norm used for its stopping check) would be the WRONG number --
   the inner product of two LOCAL slices, not of the true (distributed)
   vectors -- silently solving a well-formed but physically meaningless
   local subproblem independently on each rank.
2. Because each rank's local residual crosses scipy's convergence
   threshold at a different iteration (they're summing different subsets
   of G-vectors), ranks can decide to stop calling `matvec` at different
   iteration counts -- but `matvec` is a collective the whole
   `pwgrp_comm` group must call together the same number of times. A rank
   that stops early leaves the others waiting forever on that collective:
   a real deadlock, confirmed directly (a 10-step SplitOper CH4 run that
   completes in ~2s serially spun at 100% CPU on every rank for 90+
   seconds under ``mpirun -np 2`` without completing).

This implementation performs every inner product via `_global_vdot`, which
Allreduces across `gkspc.pwgrp_comm` whenever `gkspc` is distributed (a
no-op reducing to a plain `np.vdot` otherwise), so every rank computes
IDENTICAL scalar coefficients and takes IDENTICAL convergence decisions at
every iteration -- both correctness and lockstep termination follow from
that. For the non-distributed case this is a plain, ordinary
unpreconditioned BiCGSTAB (Van der Vorst, 1992).
"""
from __future__ import annotations

__all__ = ["bicgstab_dist"]

from typing import Callable

import numpy as np


def _global_vdot(gkspc, a: np.ndarray, b: np.ndarray) -> complex:
    val = np.vdot(a, b)
    if hasattr(gkspc, "pwgrp_comm"):
        val = gkspc.pwgrp_comm.allreduce(val)
    return val


def bicgstab_dist(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    x0: np.ndarray,
    gkspc,
    tol: float = 1e-10,
    maxiter: int = 300,
) -> tuple[np.ndarray, int]:
    """Solves ``matvec(x) = rhs`` for `x` via unpreconditioned BiCGSTAB,
    starting from `x0`. `matvec` may itself be a collective operation
    across `gkspc.pwgrp_comm` (e.g. involving `to_r()`/`to_g()`), since
    every inner product here is computed via `_global_vdot`, keeping every
    rank's iterates and stopping decision identical.

    Returns ``(x, info)`` with ``info=0`` on convergence to relative
    tolerance `tol` (measured against ``norm(rhs)``) within `maxiter`
    iterations, and ``info=1`` otherwise -- close enough to
    `scipy.sparse.linalg.bicgstab`'s convention for the call sites here,
    which only check ``info != 0``. A `rhs` that `x0` already solves
    exactly (e.g. a zero potential/Hamiltonian) is handled by the ordinary
    initial-residual check below, with no special-casing needed.
    """
    x = x0.copy()
    r = rhs - matvec(x)

    bnorm = np.sqrt(_global_vdot(gkspc, rhs, rhs).real)
    if bnorm == 0.0:
        bnorm = 1.0
    if np.sqrt(_global_vdot(gkspc, r, r).real) <= tol * bnorm:
        return x, 0

    r_hat = r.copy()
    rho = alpha = omega = 1.0 + 0.0j
    v = np.zeros_like(x)
    p = np.zeros_like(x)

    for _ in range(maxiter):
        rho_new = _global_vdot(gkspc, r_hat, r)
        if rho_new == 0.0 or omega == 0.0:
            return x, 1
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = matvec(p)
        denom = _global_vdot(gkspc, r_hat, v)
        if denom == 0.0:
            return x, 1
        alpha = rho_new / denom
        h = x + alpha * p
        s = r - alpha * v
        if np.sqrt(_global_vdot(gkspc, s, s).real) <= tol * bnorm:
            return h, 0

        t = matvec(s)
        tt = _global_vdot(gkspc, t, t).real
        omega = _global_vdot(gkspc, t, s) / tt if tt != 0.0 else 0.0

        x = h + omega * s
        r = s - omega * t
        if np.sqrt(_global_vdot(gkspc, r, r).real) <= tol * bnorm:
            return x, 0
        if omega == 0.0:
            return x, 1
        rho = rho_new

    return x, 1
