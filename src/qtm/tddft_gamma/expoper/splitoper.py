__all__ = ["SplitOper"]
import numpy as np
from qtm.containers.field import FieldRType
from qtm.containers.wavefun import get_WavefunG
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.pseudo.nloc import NonlocGenerator
from scipy.linalg import expm, block_diag
from scipy.sparse.linalg import LinearOperator, gmres
from scipy import __version__ as sc_version

from .base import TDExpOperBase

# scipy renamed gmres's 'tol' kwarg to 'rtol' in 1.12.0 (deprecating 'tol'),
# then dropped 'tol' entirely in 1.14.0 -- see nloc.py for the same pattern
# applied to the sph_harm/sph_harm_y rename.
_GMRES_TOL_KWARG = "rtol" if int(str(sc_version).split(".")[1]) >= 14 else "tol"


class SplitOper(TDExpOperBase):
    __slots__ = ["l_exp_dij_halfstep"]

    # Cayley-transform linear solve tolerances for 'oper_vloc' (see its
    # docstring for why a direct exp(-i*dt*V(r)) is not used by default).
    VLOC_SOLVER_TOL = 1e-10
    VLOC_SOLVER_MAXITER = 300

    def __init__(
        self,
        gkspc: GkSpace,
        is_spin: int,
        is_noncolin: bool,
        vloc: FieldRType,
        l_nloc: list[NonlocGenerator],
        time_step: float,
        vloc_method: str = "cayley",
    ):
        """
        Parameters
        ----------
        vloc_method : {'cayley', 'exponential'}, default='cayley'
            How `oper_vloc` applies the local-potential half/full step.

            'cayley' solves a Cayley transform (an iterative linear solve
            per band) that is EXACTLY unitary for any time step and any
            basis size -- see `oper_vloc`'s docstring for why the
            alternative is not unitary in general. Recommended default,
            especially whenever `gkspc`'s cutoff is only modestly larger
            than what the local potential's own spatial variation needs
            (the usual case for a real ionic pseudopotential).

            'exponential' applies exp(-i*dt*V(r)) directly by a single
            real-space multiplication (a plain FFT round trip, no
            iterative solve) -- much cheaper per call, and effectively
            exact whenever the plane-wave cutoff (`gkspc`'s ``ecutwfn``)
            is high enough that V(r)'s own Fourier content is already
            fully resolved within `gkspc`'s own G-sphere (so there is
            nothing left for the real-space multiplication to scatter
            outside the truncated basis and lose on projecting back). At
            a LOW-to-moderate cutoff it is measurably non-unitary and its
            single-step accuracy degrades from the expected O(dt^3) to
            O(dt^2) -- confirmed to cause outright divergence in real CH4
            TDDFT runs at ``ecutwfn`` values otherwise sufficient for
            'taylor'/`TaylorExp`. Use only once you have checked that your
            own ``ecutwfn`` is large enough for your own local potential.
        """
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc, time_step)

        if vloc_method not in ("cayley", "exponential"):
            raise ValueError(
                "'vloc_method' must be 'cayley' or 'exponential'. "
                f"got {vloc_method!r}"
            )
        self.vloc_method = vloc_method

        self.l_vkb_dij = []
        self.vnl_diag = 0
        for nloc in l_nloc:
            vkb, dij, vkb_diag = nloc.gen_vkb_dij(self.gkspc)
            print(type(vkb))
            self.l_vkb_dij.append((vkb, dij))
            self.vnl_diag += vkb_diag

        self.l_exp_dij_halfstep = []
        for vkb, dij in self.l_vkb_dij:
            ovl = vkb.vdot(vkb)
            self.l_exp_dij_halfstep.append(
                np.linalg.inv(ovl)
                @ (
                    expm(-0.5j * self.time_step * (ovl @ dij))
                    - np.identity(dij.shape[0])
                )
            )

        self.oper_exp_ke_gk_halfstep = np.exp(-0.5j * self.time_step * self.ke_gk)
        self.oper_exp_vloc_r_fullstep = None

    def update_vloc(self, vloc: FieldRType):
        super().update_vloc(vloc)  # stores self.vloc = vloc.copy(); used by 'cayley'
        # Also (cheaply) keep the direct-exponential factor current, used by
        # 'exponential' -- see 'vloc_method' and 'oper_vloc'.
        fac = -1j * self.time_step * np.prod(self.gkspc.grid_shape)
        # The line below must contain a factor 1/ np.prod(self.gkspc.grid_shape)
        # But since the wfn's are normalized at each step it is skipped
        self.oper_exp_vloc_r_fullstep = np.exp(fac * self.vloc.data.ravel())

    def _vloc_linear(self, x: np.ndarray) -> np.ndarray:
        """Applies the LINEAR local-potential operator
        ``x -> to_g(vloc(r) * to_r(x))`` to a single-band G-space coefficient
        vector ``x`` -- exactly the local-potential piece of
        `qtm.dft.ksham.KSHam.h_psi`, reused here on its own so it can be fed
        to an iterative solver for the Cayley transform in `oper_vloc`.
        """
        psi = get_WavefunG(self.gkspc, 1).empty(())
        psi.data[:] = x
        psi_r = psi.to_r()
        psi_r *= self.vloc.data.ravel()
        return psi_r.to_g().data

    def oper_ke(self, l_prop_psi: list[KSWfn]):
        np.multiply(
            self.oper_exp_ke_gk_halfstep, l_prop_psi[0].evc_gk, out=l_prop_psi[0].evc_gk
        )

    def oper_nl(self, l_prop_psi: np.ndarray, reverse: bool):
        l_ikb = range(len(self.l_vkb_dij))
        if reverse:
            l_ikb = reversed(l_ikb)
        for ikb in l_ikb:
            vkb, dij = self.l_vkb_dij[ikb]
            proj = vkb.vdot(l_prop_psi[0].evc_gk)
            l_prop_psi[0].evc_gk += (self.l_exp_dij_halfstep[ikb] @ proj).T @ vkb

    def oper_vloc(self, l_prop_psi: list[KSWfn]):
        """Applies the full-step local-potential propagator, via whichever
        of the two implementations below `self.vloc_method` selects (see
        the `vloc_method` constructor parameter for the tradeoff)."""
        if self.vloc_method == "exponential":
            self._oper_vloc_exponential(l_prop_psi)
        else:
            self._oper_vloc_cayley(l_prop_psi)

    def _oper_vloc_exponential(self, l_prop_psi: list[KSWfn]):
        r"""``vloc_method='exponential'``: applies
        :math:`\exp(-i \cdot dt \cdot V(r))` directly, by a single
        real-space multiplication (one FFT round trip, no iterative
        solve).

        This computes an EXACT unitary phase multiplication on the full
        (untruncated) real-space grid, but the map from the truncated
        plane-wave basis (``gkspc``, ``size_g`` G-vectors) back to itself
        is then the COMPRESSION of that unitary operator onto a subspace
        it does not leave invariant (multiplying by a spatially-varying
        V(r) genuinely scatters amplitude to G-vectors outside ``gkspc``,
        which gets silently discarded on projecting back) -- and the
        compression of a unitary operator onto a non-invariant subspace is
        not itself unitary in general. Verified directly (see
        '../../../tests/tddft_tests/test_expoper.py'): the resulting
        one-step propagator's own dense matrix had
        ``max|B^dagger B - I|`` of order ``dt^2``, growing with the time
        step, degrading the expected O(dt^3)-per-step Strang-splitting
        accuracy to O(dt^2) and causing outright divergence at moderate
        'dt' in a real CH4 TDDFT run -- unless ``gkspc``'s cutoff is high
        enough that V(r)'s own spatial variation is already fully resolved
        within it, in which case there is nothing left to scatter outside
        the truncated basis and this reduces to the exact answer at a
        fraction of the cost of `_oper_vloc_cayley`. See `vloc_method`.
        """
        psi_r = l_prop_psi[0].evc_gk.to_r()
        psi_r *= self.oper_exp_vloc_r_fullstep
        l_prop_psi[0].evc_gk[:] = psi_r.to_g()[:]

    def _oper_vloc_cayley(self, l_prop_psi: list[KSWfn]):
        r"""``vloc_method='cayley'`` (the default): applies the full-step
        local-potential propagator :math:`\exp(-i \cdot dt \cdot V(r))` via
        its Cayley transform,

        .. math::
            \psi_{new} = (1 + i\frac{dt}{2}V)^{-1}(1 - i\frac{dt}{2}V)\psi_{old},

        rather than a direct real-space exponential (`_oper_vloc_exponential`).

        The Cayley transform of a HERMITIAN operator is EXACTLY unitary in
        ANY finite-dimensional representation of it, truncated or not --
        unlike direct exponentiation, it never needs the working subspace to
        be invariant, because both the numerator and denominator act
        entirely within that subspace rather than exponentiating on a larger
        space and projecting down afterwards. Since 'V(r)' is real, the
        linear (untruncated-in-real-space, but ``gkspc``-basis-in/
        ``gkspc``-basis-out) operator ``_vloc_linear`` is Hermitian on
        ``gkspc``'s own basis, so this substitution restores exact
        unitarity for any time step and any basis size, at the cost of an
        iterative linear solve (GMRES) per band instead of a single
        pointwise multiplication.
        """
        evc = l_prop_psi[0].evc_gk
        size_g = self.gkspc.size_g
        data = evc.data.reshape(-1, size_g)

        a = 0.5 * self.time_step

        def matvec(x):
            return x + 1j * a * self._vloc_linear(x)

        oper = LinearOperator((size_g, size_g), matvec=matvec, dtype="c16")

        for iband in range(data.shape[0]):
            psi_old = data[iband].copy()
            v_psi_old = self._vloc_linear(psi_old)
            rhs = psi_old - 1j * a * v_psi_old

            # x0=psi_old already solves the system exactly whenever V==0
            # (or 'a' is negligibly small): the initial residual
            # rhs - matvec(psi_old) == -2j*a*v_psi_old is then exactly (or
            # numerically) zero, which degenerates scipy's GMRES (it
            # divides by the residual norm while building the Krylov basis)
            # -- skip the solve entirely in that case rather than let it
            # fail on a trivial system.
            if np.linalg.norm(v_psi_old) <= 1e-14 * (np.linalg.norm(psi_old) + 1e-300):
                data[iband] = psi_old
                continue

            psi_new, info = gmres(
                oper,
                rhs,
                x0=psi_old,
                atol=0.0,
                maxiter=self.VLOC_SOLVER_MAXITER,
                **{_GMRES_TOL_KWARG: self.VLOC_SOLVER_TOL},
            )
            if info != 0:
                raise RuntimeError(
                    "SplitOper.oper_vloc: the Cayley-transform linear solve "
                    f"did not converge (gmres info={info})."
                )
            data[iband] = psi_new

    def prop_psi(self, l_psi: list[KSWfn], l_prop_psi: list[KSWfn]):
        # 'l_psi'/'l_prop_psi' are plain Python lists of 'KSWfn' objects, so
        # 'l_prop_psi[:] = l_psi' was rebinding list slots to the SAME
        # 'KSWfn' objects (aliasing) instead of copying wavefunction data
        # into the caller-supplied 'l_prop_psi' objects -- reachable via
        # qtm.tddft_gamma.prop.etrs.prop_step whenever 'tddft_exp_method' is
        # 'splitoper' (etrs doesn't require 'tddft_prop_method' to match).
        for psi, prop_psi in zip(l_psi, l_prop_psi):
            prop_psi.evc_gk[:] = psi.evc_gk[:]
        self.oper_ke(l_prop_psi)
        self.oper_nl(l_prop_psi, False)
        self.oper_vloc(l_prop_psi)
        self.oper_nl(l_prop_psi, True)
        self.oper_ke(l_prop_psi)
