
__all__ = ["rho_check", "rho_normalize"]
from warnings import warn

import numpy as np

from quantum_masala.core import GField

EPS5 = 1E-5
EPS10 = 1E-10


def rho_check(rho: GField):
    if not isinstance(rho, GField):
        raise ValueError(f"'rho' must be an instance of 'GField'. Got {type(rho)}")
    if rho.shape not in [(1, ), (2, )]:
        raise ValueError(f"shape of 'rho' must be (1,) or  (2,). Got {rho.shape}")


def rho_normalize(rho: GField, numel: float):
    if numel < EPS5:
        raise ValueError(f"'numel' must be positive. Got{numel}")

    rho_check(rho)
    grho = rho.gspc
    rho_r = rho.r
    numspin = rho.shape[0]

    idx_neg_r = np.nonzero(rho_r.real < 0)
    if len(idx_neg_r[0]) != 0:
        del_rho = -np.sum(rho_r[idx_neg_r]) * grho.reallat_dv
        warn("negative values found in `rho.r`.\n"
             f"Error: {del_rho}")
        rho_r = np.abs(rho_r)
        rho.r = rho_r

    rho_int = np.sum(rho_r) * grho.reallat_dv

    if rho_int < EPS10:
        raise ValueError("values in 'rho.r' too small to normalize.\n"
                         f"computed total charge = {rho_int}")
    if np.abs(rho_int - numel) > EPS5:
        warn(f"total charge renormalized from {rho_int} to {numel}")
    fac = numel / rho_int
    rho *= fac

    rho.Bcast()
    return rho
