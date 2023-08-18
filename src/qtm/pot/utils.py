# from __future__ import annotations
__all__ = ['check_rho', 'fieldg_grad', 'fieldg_div']

import numpy as np
from qtm.containers import FieldG


def check_rho(rho: FieldG):
    if not isinstance(rho, FieldG):
        raise TypeError("'rho' must be a 'FieldG' instance. "
                        f"got type {type(rho)}")

    if rho.shape not in [(1, ), (2, )]:
        raise ValueError("'shape' of 'rho' must be either (1, ) or (2, ). "
                         f"got rho.shape = {rho.shape}")


def fieldg_grad(field_g: FieldG) -> FieldG:
    if not isinstance(field_g, FieldG):
        raise TypeError("'field' must be a 'FieldG' instance. "
                        f"got type {type(field_g)}.")
    gspc = field_g.gspc
    gfield = field_g.to_g()
    grad_g = np.expand_dims(gfield.data, axis=-2) * 1j * gspc.g_cart
    return FieldG(gspc, grad_g)


def fieldg_div(field_g: FieldG, axis: int = -1) -> FieldG:
    if not isinstance(field_g, FieldG):
        raise TypeError("'field' must be a 'FieldG' instance. "
                        f"got type {type(field_g)}.")
    if not isinstance(axis, int):
        raise TypeError("'axis' must be a integer. "
                        "got type {type(axis)}.")
    if field_g.shape[axis] != 3:
        raise ValueError("shape of 'field' along given 'axis' must be 3. "
                         f"got field.shape = {field_g.shape}, "
                         f"axis = {axis}")

    gspc = field_g.gspc
    div_g = np.sum(field_g.data * 1j * gspc.g_cart, axis=axis - 1)
    return FieldG(gspc, div_g)

