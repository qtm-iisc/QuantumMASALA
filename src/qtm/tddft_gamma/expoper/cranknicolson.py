r"""Crank-Nicolson propagator for the time-dependent Kohn-Sham equations.

Unlike `qtm.tddft_gamma.expoper.splitoper.SplitOper`, which factorizes
:math:`\exp(-i \cdot dt \cdot H)` into separate kinetic/local/nonlocal
sub-steps (Strang splitting), `CrankNicolson` applies the Cayley transform
directly to the FULL Hamiltonian :math:`H = T + V_{loc} + V_{nl}` (via
`qtm.dft.ksham.KSHam.h_psi`) in a single step:

.. math::
    \psi(t + dt) = (1 + i\frac{dt}{2}H)^{-1}(1 - i\frac{dt}{2}H)\psi(t)

This has two consequences relative to `SplitOper`:

* No operator-splitting error at all (the leading commutator term
  ``[T, V_loc + V_nl]`` that Strang splitting introduces is entirely
  absent here). Both this and `SplitOper` are correctly O(dt^3) locally,
  though -- the splitting commutator this avoids is just one contribution
  to `SplitOper`'s overall error constant, not the only one (its own
  per-substep Cayley/Pade approximations contribute too), so this is not
  guaranteed to be more accurate in every case: checked directly in
  '../../../tests/tddft_tests/test_expoper.py' against the same synthetic
  potential used for `SplitOper`'s own accuracy test, this propagator's
  ``err/dt^3`` constant came out roughly 25x LARGER than `SplitOper`'s for
  that particular system. Which one is more accurate for a given `dt` is
  system-dependent; what this propagator reliably avoids is the splitting
  error specifically, not error in general.
* Exactly unitary for the SAME reason `SplitOper.oper_vloc`'s Cayley
  transform is (see its docstring): the Cayley transform of a Hermitian
  operator is exactly unitary in ANY finite-dimensional representation of
  it, so this holds regardless of how many G-vectors `gkspc` retains.

Solver choice: :math:`A = 1 + i\frac{dt}{2}H` is built from a Hermitian
`H`, but `A` itself is NOT Hermitian (:math:`A^\dagger = 1 -
i\frac{dt}{2}H \neq A`) -- so plain Conjugate Gradient (which requires a
Hermitian *positive-definite* matrix) does not apply directly. `A` IS
NORMAL, though (:math:`A^\dagger A = 1 + \frac{dt^2}{4}H^2 = AA^\dagger`,
since `H` commutes with itself), sharing `H`'s own eigenvectors with
eigenvalues :math:`1 + i\frac{dt}{2}\lambda_k`. `scipy.sparse.linalg.gmres`
(a general, non-Hermitian solver) works, but its Krylov basis grows and
must be fully re-orthogonalized every iteration -- unnecessarily expensive
for a system this well-structured, since it can never distinguish that a
much cheaper *short-recurrence* method (fixed cost per iteration,
regardless of how many iterations are needed) would do just as well here.
BiCGSTAB is exactly that: it needs no explicit Hermitian symmetry
(so it works on `A` directly, no restructuring needed), only two `matvec`s
with `A` per iteration and no growing basis to store, and is the standard
practical choice for shifted-Hermitian/normal systems like this one in the
Crank-Nicolson-for-Schrodinger-equation literature -- this is what both
`CrankNicolson` and `SplitOper.oper_vloc`'s Cayley transform use, via the
distribution-aware `qtm.tddft_gamma.expoper._bicgstab.bicgstab_dist` (see
its module docstring for why `scipy.sparse.linalg.bicgstab` itself cannot
be used directly once `gkspc` is an MPI-distributed `DistGkSpace`).
"""
__all__ = ["CrankNicolson"]

import numpy as np

from qtm.containers.field import FieldRType
from qtm.containers.wavefun import get_WavefunG
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.logger import qtmlogger
from qtm.pseudo.nloc import NonlocGenerator

from ._bicgstab import bicgstab_dist
from .base import TDExpOperBase


class CrankNicolson(TDExpOperBase):
    __slots__ = []

    SOLVER_TOL = 1e-10
    SOLVER_MAXITER = 300

    def __init__(
        self,
        gkspc: GkSpace,
        is_spin: int,
        is_noncolin: bool,
        vloc: FieldRType,
        l_nloc: list[NonlocGenerator],
        time_step: float,
    ):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc, time_step)

    def _h_linear(self, x: np.ndarray) -> np.ndarray:
        """Applies the FULL Hamiltonian (kinetic + local + nonlocal, via
        `KSHam.h_psi`) to a single-band G-space coefficient vector `x`."""
        WavefunG = get_WavefunG(self.gkspc, 1)
        psi = WavefunG.empty(())
        psi.data[:] = x
        hpsi = WavefunG.empty(())
        self.h_psi(psi, hpsi)
        return hpsi.data

    def prop_psi(self, l_psi_in: list[KSWfn], l_psi_out: list[KSWfn]):
        """Propagates `l_psi_in` by one full time step via the Cayley
        transform of the full Hamiltonian, writing the result into
        `l_psi_out` (which may be the same object(s) as `l_psi_in`, or
        different ones -- either way, `l_psi_in` is left untouched)."""
        if self.is_noncolin:
            qtmlogger.warning(
                "CrankNicolson.prop_psi(): is_noncolin not implemented yet."
            )
            return

        a = 0.5 * self.time_step
        size_g = self.gkspc.size_g

        def matvec(x):
            return x + 1j * a * self._h_linear(x)

        for idxspin in range(1 + self.is_spin * (not self.is_noncolin)):
            if self.is_spin:
                self.set_idxspin(idxspin)

            # Snapshot the input before writing anything into the output,
            # since 'l_psi_in' and 'l_psi_out' may be the same object(s).
            data_in = l_psi_in[idxspin].evc_gk.data.reshape(-1, size_g).copy()
            l_psi_out[idxspin].evc_gk[:] = l_psi_in[idxspin].evc_gk[:]
            data_out = l_psi_out[idxspin].evc_gk.data.reshape(-1, size_g)

            for iband in range(data_in.shape[0]):
                psi_old = data_in[iband]
                rhs = psi_old - 1j * a * self._h_linear(psi_old)

                psi_new, info = bicgstab_dist(
                    matvec,
                    rhs,
                    psi_old,
                    self.gkspc,
                    tol=self.SOLVER_TOL,
                    maxiter=self.SOLVER_MAXITER,
                )
                if info != 0:
                    raise RuntimeError(
                        "CrankNicolson.prop_psi: the Cayley-transform linear "
                        f"solve did not converge (bicgstab info={info})."
                    )
                data_out[iband] = psi_new
