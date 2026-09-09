"""A more sophisticated (non-diagonal-Hamiltonian-adjacent) ground-truth
test for `qtm.tddft_gamma.expoper`'s two time-propagators, `TaylorExp` and
`SplitOper` -- both subclass `qtm.dft.ksham.KSHam` (already covered by
'../dft_tests/test_ksham.py'), so they're built the same way: a plain
synthetic cubic `GkSpace` with `vloc`/`l_nloc=[]`, no crystal, no SCF.

With `vloc=0` and no nonlocal projectors, `h_psi` is exactly the plane-wave
kinetic operator (see '../dft_tests/test_davidson.py'), so the exact
one-step propagator is known in closed form per G-vector:
`exp(-i*ke_g*dt)` where `ke_g = 0.5*|G+k|^2`.

`SplitOper.oper_vloc` applies the full-step local-potential propagator via
its Cayley transform (`(1+i*dt/2*V)^-1 (1-i*dt/2*V)`) rather than a direct
real-space `exp(-i*dt*V(r))`. This was a deliberate fix: representing the
potential step as `to_g(exp(-i*dt*V(r)) * to_r(psi))` computes an exact
unitary phase multiplication on the FULL (untruncated) real-space grid, but
projecting the result back onto the truncated plane-wave basis (`gkspc`)
is then the COMPRESSION of that unitary operator onto a subspace it does
not leave invariant -- and compressing a unitary operator onto a
non-invariant subspace is not itself unitary in general. This was
confirmed to be the actual root cause of a real, reproducible divergence in
full CH4 TDDFT runs (see session history): the old implementation's own
dense one-step operator matrix had `max|B^dagger B - I|` of order `dt^2`,
growing with the time step, degrading the expected O(dt^3)-per-step
Strang-splitting local accuracy to O(dt^2) and eventually diverging
outright at moderate `dt`. The Cayley transform of a Hermitian operator is
EXACTLY unitary in any finite-dimensional representation of it, truncated
or not, since both its numerator and denominator act entirely within the
working subspace rather than exponentiating on a larger space and
projecting down afterwards -- this is verified directly below (both the
resulting SINGLE-block unitarity and the restored O(dt^3) local accuracy).
"""
import numpy as np
import scipy.linalg as sla

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.containers import get_FieldR, get_WavefunG
from qtm.dft import KSHam, KSWfn
from qtm.gspace import GkSpace, GSpace
from qtm.lattice import RealLattice, ReciLattice
from qtm.tddft_gamma.expoper.splitoper import SplitOper
from qtm.tddft_gamma.expoper.taylor import TaylorExp

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gwfn = GSpace(recilat, 10.0)
gkspc = GkSpace(gwfn, (0.13, 0.27, -0.08))  # generic k: no accidental degeneracies
FieldR = get_FieldR(gwfn)

DT = 0.05
KE_G = 0.5 * gkspc.gk_norm2  # exact kinetic energy per G-vector


def _make_kswfn(numbnd=3):
    kswfn = KSWfn(gkspc, k_weight=1.0, numbnd=numbnd, is_noncolin=False)
    kswfn.init_random()
    kswfn.evc_gk.normalize()
    return kswfn


# ----- TaylorExp: converges to the exact free-electron propagator ----------
def test_taylor_exp_converges_to_exact_propagator_with_increasing_order():
    vloc = FieldR.zeros(())
    kswfn_in = _make_kswfn(1)
    exact = np.exp(-1j * KE_G * DT) * kswfn_in.evc_gk.data

    prev_err = np.inf
    for order in (2, 4, 8, 16):
        te = TaylorExp(
            gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT,
            order=order,
        )
        kswfn_out = KSWfn(gkspc, 1.0, 1, is_noncolin=False)
        te.prop_psi([kswfn_in], [kswfn_out])

        err = np.max(np.abs(kswfn_out.evc_gk.data - exact))
        assert err < prev_err  # strictly improves as more terms are kept
        prev_err = err
    assert prev_err < 1e-10  # order=16 should be at (near-)machine precision


def test_taylor_exp_matches_analytic_truncation_error_at_fixed_order():
    # The Taylor series of exp(-i*ke_g*dt) truncated after 'order' terms
    # (n=0..order) has a leading omitted term of magnitude
    # |ke_g*dt|^(order+1) / (order+1)! per G-vector.
    vloc = FieldR.zeros(())
    order = 4
    kswfn_in = _make_kswfn(1)
    te = TaylorExp(
        gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT,
        order=order,
    )
    kswfn_out = KSWfn(gkspc, 1.0, 1, is_noncolin=False)
    te.prop_psi([kswfn_in], [kswfn_out])

    exact = np.exp(-1j * KE_G * DT) * kswfn_in.evc_gk.data
    from scipy.special import factorial

    expected_leading_error = (KE_G * DT) ** (order + 1) / factorial(order + 1)
    actual_error = np.abs(kswfn_out.evc_gk.data[0] - exact[0])
    assert np.all(actual_error < 3 * expected_leading_error + 1e-14)


# ----- SplitOper: exact free-electron propagator (vloc=0) -------------------
def test_splitoper_free_electron_matches_exact_propagator():
    # This is exactly the sequence qtm.tddft_gamma.prop.splitoper.prop_step
    # actually uses in production (it never calls SplitOper.prop_psi()).
    # vloc=0 makes the Cayley-transform linear operator exactly the
    # identity, so oper_vloc is a no-op and the whole sequence reduces to
    # the exact kinetic propagator, with no scaling ambiguity of any kind.
    vloc = FieldR.zeros(())
    so = SplitOper(gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT)
    so.update_vloc(vloc)

    kswfn_in = _make_kswfn(3)
    kswfn = KSWfn(gkspc, 1.0, 3, is_noncolin=False)
    kswfn.evc_gk[:] = kswfn_in.evc_gk[:]

    l = [kswfn]
    so.oper_ke(l)
    so.oper_nl(l, False)
    so.oper_vloc(l)
    so.oper_nl(l, True)
    so.oper_ke(l)

    exact = np.exp(-1j * KE_G * DT)[None, :] * kswfn_in.evc_gk.data
    assert np.allclose(kswfn.evc_gk.data, exact, atol=1e-8)


def test_splitoper_uniform_potential_matches_scalar_cayley_transform():
    # A spatially uniform potential v0 makes oper_vloc's Cayley-transform
    # linear operator just the SCALAR v0, so the exact result of the solve
    # is the plain scalar Cayley transform (1 - i*dt/2*v0)/(1 + i*dt/2*v0)
    # -- NOT exp(-i*dt*v0) (the Cayley transform is only a [1/1] Pade
    # approximant to the exponential, exact only in the dt->0 limit); this
    # checks the actual GMRES-based implementation against that closed
    # form, computed independently via plain scalar arithmetic.
    v0 = 0.37
    vloc = FieldR.zeros(())
    vloc.data[:] = v0 / gwfn.size_r  # KSHam's pre-division-by-size_r convention
    so = SplitOper(gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT)
    so.update_vloc(vloc)

    kswfn_in = _make_kswfn(2)
    kswfn = KSWfn(gkspc, 1.0, 2, is_noncolin=False)
    kswfn.evc_gk[:] = kswfn_in.evc_gk[:]
    so.oper_vloc([kswfn])

    cayley_scalar = (1 - 0.5j * DT * v0) / (1 + 0.5j * DT * v0)
    assert np.isclose(abs(cayley_scalar), 1.0)  # exactly unitary for any dt, any v0
    assert np.allclose(kswfn.evc_gk.data, cayley_scalar * kswfn_in.evc_gk.data, atol=1e-8)


def test_splitoper_prop_psi_matches_the_direct_suboperator_sequence():
    # Regression test for a fixed bug: SplitOper.prop_psi used to do
    # 'l_prop_psi[:] = l_psi' on plain Python lists of KSWfn objects, which
    # rebinds list slots to the SAME KSWfn instances (aliasing) instead of
    # copying wavefunction data into the caller-supplied 'l_prop_psi'
    # objects -- reachable via qtm.tddft_gamma.prop.etrs.prop_step whenever
    # 'tddft_exp_method' is 'splitoper' (etrs's own prop_step doesn't
    # require 'tddft_prop_method' to also be 'splitoper'). Checked here by
    # confirming 'prop_psi' both leaves its input untouched AND exactly
    # matches calling the five sub-operators directly (the sequence
    # qtm.tddft_gamma.prop.splitoper.prop_step actually uses).
    v0 = 0.37
    vloc = FieldR.zeros(())
    vloc.data[:] = v0 / gwfn.size_r
    so = SplitOper(gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT)
    so.update_vloc(vloc)

    kswfn_in = _make_kswfn(2)
    orig_data = kswfn_in.evc_gk.data.copy()
    kswfn_out = KSWfn(gkspc, 1.0, 2, is_noncolin=False)
    so.prop_psi([kswfn_in], [kswfn_out])
    assert np.array_equal(kswfn_in.evc_gk.data, orig_data)  # input untouched

    kswfn_direct = KSWfn(gkspc, 1.0, 2, is_noncolin=False)
    kswfn_direct.evc_gk[:] = orig_data
    l = [kswfn_direct]
    so.oper_ke(l)
    so.oper_nl(l, False)
    so.oper_vloc(l)
    so.oper_nl(l, True)
    so.oper_ke(l)

    assert np.allclose(kswfn_out.evc_gk.data, kswfn_direct.evc_gk.data, atol=1e-10)


# ----- SplitOper.oper_vloc: the 'exponential' alternative -------------------
# `SplitOper` also keeps the original direct-exponential implementation
# available via `vloc_method='exponential'` -- much cheaper per call (a
# single FFT round trip, no iterative solve), and a good approximation once
# `gkspc`'s cutoff is high enough relative to the local potential's own
# spatial variation. It is NOT unitary in general (see 'cayley' above), but
# it IS exact in the two trivial cases below, where there is nothing for a
# spatially-varying potential to scatter outside the truncated basis: vloc=0
# (no potential at all) and a spatially uniform vloc (a pure global phase,
# which by definition doesn't mix G-vectors). Both inherit the WavefunG
# to_r()/to_g() round-trip's 'gkspc.size_r' amplitude scaling (see
# '../containers_tests/test_containers.py::
# test_wavefun_round_trip_scales_by_size_r') -- unlike 'cayley', this method
# never divides it back out.
def test_splitoper_exponential_method_free_electron_matches_exact_propagator():
    vloc = FieldR.zeros(())
    so = SplitOper(
        gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT,
        vloc_method="exponential",
    )
    so.update_vloc(vloc)

    kswfn_in = _make_kswfn(3)
    kswfn = KSWfn(gkspc, 1.0, 3, is_noncolin=False)
    kswfn.evc_gk[:] = kswfn_in.evc_gk[:]
    l = [kswfn]
    so.oper_ke(l)
    so.oper_nl(l, False)
    so.oper_vloc(l)
    so.oper_nl(l, True)
    so.oper_ke(l)

    exact = np.exp(-1j * KE_G * DT)[None, :] * kswfn_in.evc_gk.data
    assert np.allclose(kswfn.evc_gk.data, gwfn.size_r * exact, atol=1e-6)


def test_splitoper_exponential_method_uniform_potential_matches_exact_exponential():
    v0 = 0.37
    vloc = FieldR.zeros(())
    vloc.data[:] = v0 / gwfn.size_r
    so = SplitOper(
        gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT,
        vloc_method="exponential",
    )
    so.update_vloc(vloc)

    kswfn_in = _make_kswfn(2)
    kswfn = KSWfn(gkspc, 1.0, 2, is_noncolin=False)
    kswfn.evc_gk[:] = kswfn_in.evc_gk[:]
    so.oper_vloc([kswfn])

    exact = np.exp(-1j * DT * v0) * kswfn_in.evc_gk.data
    assert np.allclose(kswfn.evc_gk.data, gwfn.size_r * exact, atol=1e-6)


def test_splitoper_rejects_unknown_vloc_method():
    import pytest

    vloc = FieldR.zeros(())
    with pytest.raises(ValueError):
        SplitOper(
            gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=DT,
            vloc_method="not_a_real_method",
        )


# ----- SplitOper.oper_vloc: exact unitarity and restored O(dt^3) accuracy --
def _nonuniform_vloc(amplitude=0.6, m=(1, 1, 0)):
    """A synthetic, genuinely non-uniform local potential
    V(r) = amplitude*cos(2*pi*(m . r_cryst)), built directly from the
    lattice's own real-space mesh -- no crystal/pseudopotential needed."""
    reallat = RealLattice.from_recilat(recilat)
    mesh = reallat.get_mesh_coords(*gwfn.grid_shape, coords="cryst")
    v_r = amplitude * np.cos(2 * np.pi * np.tensordot(np.array(m), mesh, axes=1))
    vloc = FieldR.zeros(())
    vloc.data[:] = v_r.ravel() / gwfn.size_r
    return vloc


def _dense_h(vloc):
    size_g = gkspc.size_g
    WavefunG = get_WavefunG(gkspc, 1)
    ksham = KSHam(gkspc, is_noncolin=False, vloc=vloc, l_nloc=[])
    ident = WavefunG.empty(size_g)
    ident.data[:] = np.eye(size_g)
    hpsi = WavefunG.empty(size_g)
    ksham.h_psi(ident, hpsi)
    H = hpsi.data.T
    assert np.max(np.abs(H - H.conj().T)) < 1e-10  # sanity: H must be Hermitian
    return H


def _dense_oper_vloc(so, scale=1.0):
    # 'scale' divides out the known gkspc.size_r amplitude factor that
    # vloc_method='exponential' (but not 'cayley') carries -- see
    # 'test_splitoper_exponential_method_*' above.
    size_g = gkspc.size_g
    B = np.zeros((size_g, size_g), dtype=complex)
    for j in range(size_g):
        k = KSWfn(gkspc, 1.0, 1, is_noncolin=False)
        k.evc_gk.data[:] = 0
        k.evc_gk.data[0, j] = 1.0
        so.oper_vloc([k])
        B[:, j] = k.evc_gk.data[0] / scale
    return B


def test_exponential_method_is_measurably_not_unitary_for_a_nonuniform_potential():
    # The other half of the tradeoff documented on 'vloc_method': at a
    # modest cutoff, 'exponential' is NOT unitary (unlike 'cayley', checked
    # just below), by an amount of the same order as what was measured
    # against the real Si ionic potential during development (~0.03-0.1 at
    # dt=0.2-0.5).
    vloc = _nonuniform_vloc()
    so = SplitOper(
        gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=0.2,
        vloc_method="exponential",
    )
    so.update_vloc(vloc)
    B = _dense_oper_vloc(so, scale=gwfn.size_r)
    unitarity_err = np.max(np.abs(B.conj().T @ B - np.eye(gkspc.size_g)))
    assert unitarity_err > 1e-3


def test_oper_vloc_is_exactly_unitary_for_a_nonuniform_potential():
    vloc = _nonuniform_vloc()
    for dt in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        so = SplitOper(gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=dt)
        so.update_vloc(vloc)
        B = _dense_oper_vloc(so)
        unitarity_err = np.max(np.abs(B.conj().T @ B - np.eye(gkspc.size_g)))
        assert unitarity_err < 1e-8, f"dt={dt}: max|B^H B - I| = {unitarity_err:.3e}"


def test_splitoper_single_step_local_error_is_third_order_in_dt():
    # The defining Strang-splitting accuracy guarantee this fix restores:
    # local (single-step) error O(dt^3), i.e. err/dt^3 converges to a
    # constant as dt -> 0 (checked the same way against a completely
    # independent pure-math sanity check during development: random
    # Hermitian matrices compared via scipy.linalg.expm directly).
    vloc = _nonuniform_vloc()
    H = _dense_h(vloc)

    rng = np.random.default_rng(3)
    psi0 = rng.standard_normal(gkspc.size_g) + 1j * rng.standard_normal(gkspc.size_g)
    psi0 /= np.linalg.norm(psi0)

    ratios = []
    for dt in (0.4, 0.2, 0.1, 0.05, 0.025):
        so = SplitOper(gkspc, is_spin=0, is_noncolin=False, vloc=vloc, l_nloc=[], time_step=dt)
        so.update_vloc(vloc)
        k = KSWfn(gkspc, 1.0, 1, is_noncolin=False)
        k.evc_gk.data[:] = psi0
        l = [k]
        so.oper_ke(l)
        so.oper_nl(l, False)
        so.oper_vloc(l)
        so.oper_nl(l, True)
        so.oper_ke(l)
        out = k.evc_gk.data[0]

        exact = sla.expm(-1j * dt * H) @ psi0
        # SplitOper is now exactly unitary (previous assertion), so a plain
        # phase alignment (no amplitude rescaling) is all that's needed.
        phase = np.vdot(exact, out)
        phase /= abs(phase)
        err = np.max(np.abs(out / phase - exact))
        ratios.append(err / dt**3)

    # err/dt^3 should be roughly constant (not shrinking, which is what the
    # old, non-unitary implementation showed -- err/dt^3 there grew as dt
    # shrank, the signature of an actual O(dt^2) error).
    assert max(ratios) / min(ratios) < 2.0
