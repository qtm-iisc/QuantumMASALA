from __future__ import annotations

__all__ = ["check_rho", "fieldg_grad", "fieldg_div"]

import numpy as np
from qtm.containers import FieldGType

from qtm.msg_format import *


def check_rho(rho: FieldGType):
    if not isinstance(rho, FieldGType):
        raise TypeError(type_mismatch_msg("rho", rho, FieldGType))

    if rho.shape not in [(1,), (2,)]:
        raise ValueError(
            value_mismatch_msg("rho.shape", rho.shape, "either (1, ) or (2, )")
        )


def fieldg_grad(field_g: FieldGType) -> FieldGType:
    if not isinstance(field_g, FieldGType):
        raise TypeError(type_mismatch_msg("field_g", field_g, FieldGType))
    gspc = field_g.gspc
    grad_g = field_g.data[..., np.newaxis, :] * 1j * gspc.g_cart
    return type(field_g)(grad_g)


def fieldg_div(field_g: FieldGType, axis: int = -1) -> FieldGType:
    if not isinstance(field_g, FieldGType):
        raise TypeError(type_mismatch_msg("field_g", field_g, FieldGType))
    if not isinstance(axis, int):
        raise TypeError("'axis' must be a integer. " "got type {type(axis)}.")
    if field_g.shape[axis] != 3:
        raise ValueError(
            value_mismatch_msg(
                "field_g.shape[axis]", field_g.shape[axis], f"3 for axis = {axis}"
            )
        )
    gspc = field_g.gspc
    div_g = np.sum(field_g.data * (1j * gspc.g_cart), axis=axis - 1)
    return type(field_g)(div_g)
