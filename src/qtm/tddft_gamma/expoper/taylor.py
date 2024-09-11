__all__ = ['TaylorExp']
from copy import deepcopy
import numpy as np
from qtm.containers.field import FieldRType
from qtm.dft.kswfn import KSWfn
from qtm.gspace.gkspc import GkSpace
from qtm.logger import qtmlogger
from qtm.pseudo.nloc import NonlocGenerator
from qtm.tddft_gamma.expoper.base import TDExpOperBase
from scipy.linalg.blas import zaxpy


class TaylorExp(TDExpOperBase):

    __slots__ = ['order']

    def __init__(self, gkspc: GkSpace, is_spin: int, is_noncolin: bool,
                 vloc: FieldRType, l_nloc: list[NonlocGenerator],
                 time_step: float, order: int = 4):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc,
                         time_step)

        if not isinstance(order, int) or order < 1:
            raise ValueError("'order' must be a positive integer. "
                             f"got '{order}' (type {type(order)})")
        self.order = order

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
                fac *= -1j * self.time_step / (iorder + 1)
                
                # prop_psi._data += fac * h_psi._data
                # FIXME: This is a temporary fix. The private attribute _data should not be accessed directly.
                zaxpy(x=h_psi._data.reshape(-1), y=prop_psi._data.reshape(-1), a=fac)

                # psi = H^{n-1} * psi
                # Swap the pointers instead of copying the data.
                psi, h_psi = h_psi, psi

            l_psi_out[idxspin].evc_gk._data = prop_psi._data.copy()
            # l_psi_out[idxspin].evc_gk.normalize()