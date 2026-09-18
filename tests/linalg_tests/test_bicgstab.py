"""Unit tests for `qtm.linalg.bicgstab`, independent of `GkSpace` or MPI:
`bicgstab_dist`/`_global_vdot` only ever call `matvec` and
`gkspc.pwgrp_comm.allreduce` (skipped entirely when `gkspc` has no
`pwgrp_comm` attribute), so these can be checked directly against small
dense linear systems, with a trivial stand-in communicator used
specifically to exercise the "distributed" code path without real MPI.
"""
import numpy as np

from qtm.linalg.bicgstab import bicgstab_dist


def _random_normal_system(n, a, seed):
    """A dense system shaped like the one `CrankNicolson`/`SplitOper`'s
    Cayley transform actually solves: `A = I + i*a*H` for Hermitian `H`,
    real scalar `a` -- normal but not Hermitian (see `bicgstab.py`'s
    module docstring for why that rules out plain Conjugate Gradient)."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (H + H.conj().T)
    A = np.eye(n) + 1j * a * H
    rhs = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return A, rhs


class _FakeSingleRankComm:
    """Stand-in for `gkspc.pwgrp_comm`: a single-"rank" communicator whose
    `allreduce` is the identity. Exists purely so the `hasattr(gkspc,
    "pwgrp_comm")` branch (unreachable with `gkspc=None`, used everywhere
    else in these tests) is actually exercised at least once, without
    needing real MPI."""

    def allreduce(self, val):
        return val


class _FakeDistGkspc:
    pwgrp_comm = _FakeSingleRankComm()


def test_bicgstab_dist_solves_a_dense_normal_system():
    n = 30
    A, rhs = _random_normal_system(n, a=0.5, seed=0)
    x0 = np.zeros(n, dtype=complex)

    x, info = bicgstab_dist(lambda v: A @ v, rhs, x0, None, tol=1e-12, maxiter=300)

    assert info == 0
    exact = np.linalg.solve(A, rhs)
    assert np.allclose(x, exact, atol=1e-8)


def test_bicgstab_dist_handles_an_already_solved_rhs():
    # The docstring specifically calls this out as needing no special
    # casing: 'x0' already solves 'matvec(x) = rhs' exactly.
    n = 20
    A, _ = _random_normal_system(n, a=0.3, seed=1)
    x0 = np.random.default_rng(2).standard_normal(n) + 1j * np.random.default_rng(
        3
    ).standard_normal(n)
    rhs = A @ x0

    x, info = bicgstab_dist(lambda v: A @ v, rhs, x0, None, tol=1e-10, maxiter=300)

    assert info == 0
    assert np.allclose(x, x0, atol=1e-10)


def test_bicgstab_dist_converges_faster_from_a_good_initial_guess():
    n = 40
    A, rhs = _random_normal_system(n, a=0.5, seed=4)
    exact = np.linalg.solve(A, rhs)

    def counting_matvec(v, counter):
        counter[0] += 1
        return A @ v

    counter_good = [0]
    x0_good = exact + 1e-6 * (
        np.random.default_rng(5).standard_normal(n)
        + 1j * np.random.default_rng(6).standard_normal(n)
    )
    bicgstab_dist(
        lambda v: counting_matvec(v, counter_good),
        rhs,
        x0_good,
        None,
        tol=1e-10,
        maxiter=300,
    )

    counter_poor = [0]
    x0_poor = np.zeros(n, dtype=complex)
    bicgstab_dist(
        lambda v: counting_matvec(v, counter_poor),
        rhs,
        x0_poor,
        None,
        tol=1e-10,
        maxiter=300,
    )

    assert counter_good[0] < counter_poor[0]


def test_bicgstab_dist_reports_failure_rather_than_looping_forever():
    n = 20
    A, rhs = _random_normal_system(n, a=0.5, seed=7)
    x0 = np.zeros(n, dtype=complex)

    # A perfectly solvable system, but with zero iteration budget: the
    # only possible reason for not converging is the budget itself, not
    # some other pathological edge case.
    x, info = bicgstab_dist(lambda v: A @ v, rhs, x0, None, tol=1e-12, maxiter=0)

    assert info == 1
    assert np.array_equal(x, x0)  # no iterations means no change from x0


def test_bicgstab_dist_uses_the_distributed_allreduce_path_when_present():
    # Same system and answer as the plain (gkspc=None) case, but routed
    # through 'gkspc.pwgrp_comm.allreduce' at every inner product -- the
    # only way to exercise that branch without real MPI.
    n = 25
    A, rhs = _random_normal_system(n, a=0.4, seed=8)
    x0 = np.zeros(n, dtype=complex)

    x, info = bicgstab_dist(
        lambda v: A @ v, rhs, x0, _FakeDistGkspc(), tol=1e-12, maxiter=300
    )

    assert info == 0
    exact = np.linalg.solve(A, rhs)
    assert np.allclose(x, exact, atol=1e-8)
