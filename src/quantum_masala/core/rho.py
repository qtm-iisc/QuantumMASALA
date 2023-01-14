
__all__ = ["rho_check", "rho_normalize"]
from typing import Optional

import numpy as np

from quantum_masala.core import GField
from quantum_masala import pw_logger

EPS5 = 1E-4
EPS10 = 1E-10


def rho_check(rho: GField, is_spin: Optional[bool] = None):
    """Checks if input ``rho`` matches the expected type and shape for
    an electron density object.

    Parameters
    ----------
    rho : GField
        Represents the charge density of system
    is_spin : Optional[bool]
        Spin-polarized if True

    Raises
    -------
    TypeError
        If either ``rho`` is not an instance of ``GField`` or ``is_spin`` is
        neither None nor boolean
    ValueError
        If ``rho.shape`` is not ``(1, )`` nor ``(2, )``.
    """
    if not isinstance(rho, GField):
        raise TypeError("'rho' must be an instance of 'GField'. "
                        f"got {type(rho)}.")
    if is_spin is not None and not isinstance(is_spin, bool):
        raise TypeError("'is_spin' must be either None or a boolean. "
                        f"got {type(is_spin)}.")

    if is_spin is None:
        if rho.shape not in [(1, ), (2, )]:
            raise ValueError(f"'rho.shape' must be (1,) or  (2,); got {rho.shape}")
    elif rho.shape != (1 + is_spin, ):
        raise ValueError(f"'rho.shape' must be ({1 + is_spin}, ). got {rho.shape}")


def rho_normalize(rho: GField, numel: float):
    """Normalizes the electron density represented by ``rho`` to match the
    number of electrons per unit cell ``numel``

    Parameters
    ----------
    rho : GField
        Input electron density
    numel : float
        Number of electrons per unit cell. Must be positive.

    Returns
    -------
    rho : GField
        Input electron density that is normalized to contain ``numel`` number
        of electrons per unit cell

    Raises
    ------
    ValueError
        If ``numel`` is not positive or net charge in unit cell given by ``rho``
        is too small
    """
    if numel < EPS5:
        raise ValueError(f"'numel' must be positive. Got{numel}")

    rho_check(rho)
    grho = rho.gspc
    rho_r = rho.to_rfield()

    idx_neg_r = np.nonzero(rho_r.r.real < 0)
    if len(idx_neg_r[0]) != 0:
        del_rho = -np.sum(rho_r.r[idx_neg_r]) * grho.reallat_dv
        pw_logger.warn("negative values found in `rho.r`.\n"
                       f"error: {del_rho}.")
        rho_r.r[:] = np.abs(rho_r.r)

    rho_int = rho_r.integrate(axis=0)

    if rho_int < EPS10:
        raise ValueError("values in 'rho.r' too small to normalize.\n"
                         f"computed total charge = {rho_int}")
    if np.abs(rho_int - numel) > EPS5:
        pw_logger.warn(f"total charge renormalized from {rho_int} to {numel}")
    fac = numel / rho_int
    rho_r *= fac

    rho = rho_r.to_gfield()
    rho.Bcast()
    return rho
