from __future__ import annotations
__all__ = ['compute']

import numpy as np
from qtm.containers import FieldGType, FieldRType
from qtm.constants import FPI
from .utils import check_rho


def compute(rho: FieldGType) -> tuple[FieldRType, float]:
    """Computes the Hartree Potential and the corresponding interaction energy
    per unit cell for input charge.

    Parameters
    ----------
    rho : FieldG
        Input Charge density. `rho.shape` must be ``(2, )`` or ``(1, )``
    Returns
    -------
    v_hart : FieldR
        Hartree Potential. Is a scalar field as it is spin independent
    en_hart : float
        Interaction Energy per unit cell
    """
    check_rho(rho)
    gspc = rho.gspc
    rho: FieldGType = sum(rho)
    v_g = FPI * rho
    with np.errstate(divide='ignore'):
        v_g /= gspc.g_norm2
    if gspc.has_g0:
        v_g.data[..., 0] = 0

    en_hart = 0.5 * np.sum(rho.conj() * v_g)
    en_hart *= gspc.reallat_dv / np.prod(gspc.grid_shape)

    return v_g.to_r(), en_hart.real
