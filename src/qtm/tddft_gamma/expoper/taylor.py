__all__ = ["TaylorExp"]
from copy import deepcopy
from typing import Optional
import numpy as np
from qtm.containers.field import FieldRType
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.logger import qtmlogger
from qtm.pseudo.nloc import NonlocGenerator
from qtm.tddft_gamma.expoper.base import TDExpOperBase
from scipy.linalg.blas import zaxpy


class TaylorExp(TDExpOperBase):
    r"""Taylor-series propagator.

    By default (`e_ref=None`, unchanged from before this class supported
    shifting) this expands :math:`\exp(-i\,dt\,H)` directly in powers of
    `H`, i.e. around physical zero energy. For a typical DFT Hamiltonian,
    whose eigenvalues (kinetic + potential) are all far from zero, that is
    a poorly-centered expansion point -- the number of terms needed is set
    by the LARGEST eigenvalue magnitude actually present in `H`'s
    spectrum, not by how spread out that spectrum is, purely because the
    series has no way to "start closer" to where the eigenvalues actually
    are.

    Passing `e_ref` (typically :math:`\bar E = (E_{\max}+E_{\min})/2`,
    the same reference `qtm.tddft_gamma.expoper.kte.KTEExp` computes for
    its own rescaling -- see `qtm.linalg.lanczos`) factors out the
    corresponding global phase and expands the remainder around
    `H - e_ref` instead:

    .. math::
        \exp(-i\,dt\,H) = e^{-i\,dt\,e_{ref}}\,\exp(-i\,dt\,(H-e_{ref}))
        \approx e^{-i\,dt\,e_{ref}} \sum_{n=0}^{N}
        \frac{(-i\,dt)^n}{n!}(H-e_{ref})^n

    This is mathematically the same propagator (the shift is exact, not
    an approximation), just re-centered so the truncation order needed is
    now set by the spectral RANGE around `e_ref`, rather than by the
    largest eigenvalue magnitude in `H`'s raw (unshifted) spectrum.

    Kept optional and defaulting to the old behavior rather than always
    applying it: unlike `KTEExp`, this class has no built-in spectral
    estimator, and forcing a bounds computation (or requiring the caller
    to always supply one) on every use would be a behavior change for the
    existing default-`order=4`, no-shift use of this class -- the
    existing `test_expoper.py` construction relies on that being unchanged.
    """

    __slots__ = ["order", "e_ref"]

    def __init__(
        self,
        gkspc: GkSpace,
        is_spin: int,
        is_noncolin: bool,
        vloc: FieldRType,
        l_nloc: list[NonlocGenerator],
        time_step: float,
        order: int = 4,
        e_ref: Optional[float] = None,
    ):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc, time_step)

        if not isinstance(order, int) or order < 1:
            raise ValueError(
                "'order' must be a positive integer. "
                f"got '{order}' (type {type(order)})"
            )
        self.order = order
        self.e_ref = e_ref
        """Reference energy to shift `H` by before expanding (see class
        docstring); `None` reproduces the original unshifted behavior."""

    def prop_psi(self, l_psi_in: list[KSWfn], l_psi_out: list[KSWfn]):
        """
        Propagates the wavefunction using the Taylor expansion method.

        Args:
            l_psi_in (list[KSWfn]): List of input wavefunctions.
            l_psi_out (list[KSWfn]): List of wavefunctions to store the output, i.e. the propagated wavefunctions.

        Returns:
            None

        Raises:
            None

        Note:
            Ensure that vloc is updated before calling this method.

        """
        if self.is_noncolin:
            qtmlogger.warning("TaylorExp.prop_psi(): is_noncolin not implemented yet.")
            return

        for idxspin in range(1 + self.is_spin * (not self.is_noncolin)):
            if self.is_spin:
                self.set_idxspin(idxspin)

            psi = l_psi_in[idxspin].evc_gk.copy()
            """Stores H^{n-1} * psi."""

            h_psi = psi.zeros(psi.shape)
            """Stores H^{n} * psi."""

            prop_psi = l_psi_in[idxspin].evc_gk.copy()
            prop_psi._data[:] = psi._data[:]
            """Stores the final result of the propagation."""

            fac = 1
            for iorder in range(self.order):
                self.h_psi(psi, h_psi)
                if self.e_ref is not None:
                    h_psi._data -= self.e_ref * psi._data
                fac *= -1j * self.time_step / (iorder + 1)

                # prop_psi._data += fac * h_psi._data
                # FIXME: This is a temporary fix. The private attribute _data should not be accessed directly.
                zaxpy(x=h_psi._data.reshape(-1), y=prop_psi._data.reshape(-1), a=fac)

                # psi = H^{n-1} * psi
                # Swap the pointers instead of copying the data.
                psi, h_psi = h_psi, psi

            if self.e_ref is not None:
                prop_psi._data *= np.exp(-1j * self.e_ref * self.time_step)

            l_psi_out[idxspin].evc_gk._data = prop_psi._data.copy()
            # l_psi_out[idxspin].evc_gk.normalize()
