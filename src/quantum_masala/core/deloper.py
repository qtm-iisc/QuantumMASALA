__all__ = ["compute_grad", "compute_div"]
import numpy as np

from quantum_masala.core import GField


def compute_grad(field: GField):
    gspc = field.gspc
    grad_g = np.expand_dims(field.g, axis=-2) * 1j * gspc.cart
    return GField.from_array(gspc, grad_g)


def compute_div(field: GField):
    if field.shape[-1] != 3:
        raise ValueError("Input 'field' not vaild: 'shape[-1]' must be 3. "
                         f"Got '{field.shape[-1]}'")
    gspc = field.gspc
    div_g = np.sum(field.g * 1j * gspc.cart, axis=-2)
    return GField.from_array(gspc, div_g)
