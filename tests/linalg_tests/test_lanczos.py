"""Unit tests for `qtm.linalg.lanczos`, entirely independent of `GkSpace`
or any TDDFT machinery: `lanczos_steps`/`lanczos_tridiagonalize`/
`lanczos_bounds` only ever call `matvec` and `_global_vdot(gkspc, ...)`,
and `_global_vdot` reduces to a plain `np.vdot` whenever
`hasattr(gkspc, "pwgrp_comm")` is false -- true for `gkspc=None`, used
throughout below -- so these can be checked directly against small dense
matrices with known eigenvalues, no distributed G-space setup needed.
"""
import numpy as np

from qtm.linalg.lanczos import lanczos_bounds, lanczos_steps, lanczos_tridiagonalize


def _random_hermitian(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (A + A.conj().T)


def _random_vec(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return v / np.linalg.norm(v)


def test_ritz_extremes_converge_to_true_extremal_eigenvalues():
    n = 40
    H = _random_hermitian(n, seed=0)
    true_eigs = np.linalg.eigvalsh(H)
    psi0 = _random_vec(n, seed=1)

    prev_err = np.inf
    for m in (5, 10, 20, n):
        _, alphas, betas = lanczos_tridiagonalize(lambda x: H @ x, None, psi0, m)
        ritz = (
            alphas
            if alphas.size == 1
            else np.linalg.eigvalsh(
                np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
            )
        )
        err = max(abs(ritz.min() - true_eigs.min()), abs(ritz.max() - true_eigs.max()))
        assert err < prev_err or err < 1e-12
        prev_err = err
    assert prev_err < 1e-10  # full-dimension Krylov space: exact diagonalization


def test_lanczos_tridiagonalize_matches_taking_m_steps_from_lanczos_steps():
    n = 30
    H = _random_hermitian(n, seed=2)
    psi0 = _random_vec(n, seed=3)

    for m in (1, 3, 7, 15):
        Q1, alphas1, betas1 = lanczos_tridiagonalize(lambda x: H @ x, None, psi0, m)
        gen = lanczos_steps(lambda x: H @ x, None, psi0)
        for _, result in zip(range(m), gen):
            Q2, alphas2, betas2 = result
        assert len(Q1) == len(Q2)
        assert np.allclose(alphas1, alphas2)
        assert np.allclose(betas1, betas2)
        for q1, q2 in zip(Q1, Q2):
            assert np.array_equal(q1, q2)


def test_lanczos_steps_stops_immediately_on_an_exact_eigenvector():
    n = 20
    H = _random_hermitian(n, seed=4)
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = eigvecs[:, 3]  # an exact eigenvector: Krylov space is 1-dimensional

    results = list(lanczos_steps(lambda x: H @ x, None, psi0))
    assert len(results) == 1
    Q, alphas, betas = results[0]
    assert len(Q) == 1
    assert betas.size == 0
    assert np.isclose(alphas[0], eigvals[3])


def test_lanczos_bounds_return_vectors_gives_genuine_eigenvectors():
    n = 40
    H = _random_hermitian(n, seed=5)
    true_eigs = np.linalg.eigvalsh(H)
    psi0 = _random_vec(n, seed=6)

    (e_min, e_max), v_min, v_max = lanczos_bounds(
        lambda x: H @ x, None, psi0, n_iter=n, margin=0.0, return_vectors=True
    )
    assert np.isclose(e_min, true_eigs.min(), atol=1e-8)
    assert np.isclose(e_max, true_eigs.max(), atol=1e-8)

    for e, v in ((e_min, v_min), (e_max, v_max)):
        assert np.isclose(np.linalg.norm(v), 1.0)
        residual = np.linalg.norm(H @ v - e * v)
        assert residual < 1e-6


def test_lanczos_bounds_margin_pads_symmetrically_and_monotonically():
    # conv_tol is pinned to 0.0 throughout (rather than left at its
    # margin-tracking default) specifically so every call below runs the
    # SAME full n_iter iterations and thus shares the same raw (unpadded)
    # Ritz estimate -- isolating the padding math from the (separately
    # tested) early-stopping behavior, which would otherwise make a
    # larger margin also stop earlier and so change the raw estimate too.
    n = 25
    H = _random_hermitian(n, seed=7)
    psi0 = _random_vec(n, seed=8)

    e_min0, e_max0 = lanczos_bounds(
        lambda x: H @ x, None, psi0, n_iter=n, margin=0.0, conv_tol=0.0
    )
    prev_pad = 0.0
    for margin in (0.01, 0.05, 0.2):
        e_min, e_max = lanczos_bounds(
            lambda x: H @ x, None, psi0, n_iter=n, margin=margin, conv_tol=0.0
        )
        pad_lo = e_min0 - e_min
        pad_hi = e_max - e_max0
        assert pad_lo > prev_pad and pad_hi > prev_pad
        assert np.isclose(pad_lo, pad_hi)  # symmetric padding on each side
        prev_pad = pad_lo


def test_lanczos_bounds_stops_early_once_extremal_ritz_values_stabilize():
    # A well-separated spectrum: Lanczos on this converges its extremal
    # Ritz values to the true extremes within a handful of iterations, far
    # short of the n_iter cap -- so with periodic convergence checking,
    # lanczos_bounds should stop (and thus call matvec) long before
    # reaching that cap, rather than always paying for the worst case.
    n = 50
    rng = np.random.default_rng(12)
    eigs = np.concatenate([rng.uniform(-1, 1, n - 2), [-20.0, 20.0]])
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    H = (Q * eigs) @ Q.conj().T
    H = 0.5 * (H + H.conj().T)
    psi0 = _random_vec(n, seed=13)

    call_count = 0

    def counted_matvec(x):
        nonlocal call_count
        call_count += 1
        return H @ x

    n_iter_cap = 30
    (e_min, e_max) = lanczos_bounds(
        counted_matvec, None, psi0, n_iter=n_iter_cap, margin=0.01, check_every=5
    )
    assert call_count < n_iter_cap  # stopped early, did not exhaust the cap

    # The early-stopped estimate should still be about as accurate as
    # running the full cap: the padding (margin) is the only reason for
    # the two to differ at all.
    (e_min_full, e_max_full) = lanczos_bounds(
        lambda x: H @ x, None, psi0, n_iter=n_iter_cap, margin=0.01, check_every=5
    )
    assert np.isclose(e_min, e_min_full, atol=1e-6)
    assert np.isclose(e_max, e_max_full, atol=1e-6)
    assert np.isclose(e_min, -20.0, atol=0.5)
    assert np.isclose(e_max, 20.0, atol=0.5)


def test_lanczos_bounds_zero_margin_disables_early_stopping_by_default():
    # margin=0.0 with conv_tol left at its default (= margin) means "never
    # stop early" -- a relative change of exactly zero essentially never
    # occurs, so this reduces to the old fixed-n_iter behavior. Several
    # other tests in this file rely on that to get an EXACT iteration
    # count; this test makes the guarantee explicit.
    n = 30
    H = _random_hermitian(n, seed=14)
    psi0 = _random_vec(n, seed=15)

    call_count = 0

    def counted_matvec(x):
        nonlocal call_count
        call_count += 1
        return H @ x

    n_iter = 12
    lanczos_bounds(counted_matvec, None, psi0, n_iter=n_iter, margin=0.0)
    assert call_count == n_iter


def test_lanczos_bounds_from_a_localized_state_needs_far_fewer_iterations():
    # A deliberately bimodal spectrum: a cluster of eigenvalues near 0, and
    # one eigenvalue split far away -- the situation LanczosExp is built to
    # exploit for a real Hamiltonian whose kinetic-energy cutoff sets a far
    # larger E_max than a spectrally-localized wavefunction ever reaches.
    #
    # A state with EXACTLY zero analytic overlap with the far eigenvalue
    # still eventually "discovers" it after enough iterations: its
    # machine-epsilon-level rounding-error overlap gets amplified by the
    # large eigenvalue ratio the same way the power method amplifies a
    # dominant eigenvalue's component, so this is checked with a small,
    # SHARED iteration budget where that amplification hasn't caught up
    # yet -- not by claiming it never happens at any iteration count.
    n = 60
    rng = np.random.default_rng(9)
    eigs = np.concatenate([rng.uniform(-1, 1, n - 1), [50.0]])
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    H = (Q * eigs) @ Q.conj().T
    H = 0.5 * (H + H.conj().T)

    psi_localized = Q[:, : n - 1] @ _random_vec(
        n - 1, seed=10
    )  # zero overlap with the outlier
    psi_localized /= np.linalg.norm(psi_localized)
    psi_generic = _random_vec(n, seed=11)

    m_small = 6
    _, e_max_localized = lanczos_bounds(
        lambda x: H @ x, None, psi_localized, n_iter=m_small, margin=0.0
    )
    _, e_max_generic = lanczos_bounds(
        lambda x: H @ x, None, psi_generic, n_iter=m_small, margin=0.0
    )
    assert e_max_generic > 45.0  # already resolved the far eigenvalue
    assert (
        e_max_localized < 2.0
    )  # nowhere near it yet -- an accurate, tight bound instead
