from typing import Literal

import numpy as np

from .gspc import GSpace
from .fft import FFTGSpace


class DelOperator:
    def __init__(self, gspc: GSpace, fft_gspc: FFTGSpace):
        self.gspc = gspc
        self.fft_mod = fft_gspc

    def _arr_g(self, arr_in: np.ndarray, is_vec: bool = False):
        if arr_in.shape[-1] == self.gspc.numg:
            if not is_vec:
                return arr_in
            elif arr_in.shape[-2] == 3:
                return arr_in
        if arr_in.shape[-3:] == self.gspc.grid_shape:
            if not is_vec:
                return self.fft_mod.r2g(arr_in)
            elif arr_in.shape[-4] == 3:
                return self.fft_mod.r2g(arr_in)
        if is_vec:
            raise ValueError(
                f"'arr_in' has incompatible shape. Expected {(..., 3, self.gspc.numg)} "
                f"or {(..., 3, *self.gspc.grid_shape)}, got {arr_in.shape}"
            )
        else:
            raise ValueError(
                f"'arr_in' has incompatible shape. Expected {(..., self.gspc.numg)} "
                f"or {(..., *self.gspc.grid_shape)}, got {arr_in.shape}"
            )

    def grad(self, arr_in: np.ndarray, spc_typ: Literal["r", "g"] = "r"):
        arr_g = self._arr_g(arr_in)
        arr_g = np.expand_dims(arr_g, axis=-2)
        grad_g = arr_g * 1j * self.gspc.cart

        if spc_typ == "r":
            return self.fft_mod.g2r(grad_g)
        elif spc_typ == "g":
            return grad_g
        else:
            raise ValueError(
                f"invalid value for 'spc_typ'. Expected 'r' or 'g', got {spc_typ}"
            )

    def div(self, arr_in: np.ndarray, spc_typ: Literal["r", "g"] = "r"):
        arr_g = self._arr_g(arr_in, is_vec=True)
        arr_g *= 1j * self.gspc.cart
        div_g = np.sum(arr_g, axis=-2)

        if spc_typ == "r":
            return self.fft_mod.g2r(div_g)
        elif spc_typ == "g":
            return arr_g
        else:
            raise ValueError(
                f"invalid value for 'spc_typ'. Expected 'r' or 'g', got {spc_typ}"
            )

    def curl(self, arr_in: np.ndarray, spc_typ: Literal["r", "g"] = "r"):
        raise NotImplemented("curl operator not implemented")
