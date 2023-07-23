# from __future__ import annotations
__all__ = ['compute']

from qtm.containers import FieldR, FieldG
from qtm.constants import FPI
from .utils import check_rho


def compute(rho: FieldG) -> FieldR:
    check_rho(rho)
    gspc = rho.gspc
    v_g: FieldG = FPI * sum(rho)
    if gspc.has_g0:
        v_g.g[0] = 0
        v_g.g[1:] /= gspc.g_norm2[1:]
    else:
        v_g /= gspc.g_norm2[1:]

    return v_g.to_fieldr()
