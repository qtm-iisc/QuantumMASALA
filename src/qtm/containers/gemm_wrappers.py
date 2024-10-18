from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
__all__ = ["get_zgemm", "ZGEMMWrapper"]

from sys import version_info
from functools import lru_cache
import numpy as np

from qtm.config import CUPY_INSTALLED, NDArray


if version_info[1] >= 8:
    from typing import Protocol

    class ZGEMMWrapper(Protocol):
        """Wraps ZGEMM function calls defined in Linear Algebra libraries

        ZGEMM calls are preferred over `numpy.matmul` and its analogues in
        situations where the operands require (hermitian) transposition
        before matrix multiplying. Using matmul in such cases would require
        creating extra array to store the intermediate (hermitian) transpose,
        wheras ZGEMM can save memory on such operations. ZGEMM also saves on
        creating an intermediate matmul result array before scaling and
        adding to an output array.

        NOTE: The wrapper does very little checks on the input arguments and
        might not behave as expected in some situations. Test thorougly before
        using the wrapper.
        """

        def __call__(
            self,
            a: NDArray,
            b: NDArray,
            trans_a: Literal[0, 1, 2] = 0,
            trans_b: Literal[0, 1, 2] = 0,
            alpha: complex = 1.0,
            out: NDArray | None = None,
            beta: complex = 0.0,
        ) -> NDArray:
            """Implements C := alpha*op( A )*op( B ) + beta*C, where op( X ) is
            one of: op( X ) = X or op( X ) = X**T or op( X ) = X**H

            Array arguments are expected to be in Fortran contiguous order.
            `trans_a` and `trans_b` take 0, 1 or 2 where:

            * 0 -> op( X ) = X
            * 1 -> op( X ) = X**T (transpose)
            * 2 -> op( X ) = X**H (transpose conjugate)

            Parameters
            ----------
            a : NDArray
            b : NDArray
            trans_a: Literal[0, 1, 2], default=0
            trans_b: Literal[0, 1, 2], default=0
            alpha: complex, default=1.0
            out: Optional[NDArray], default=None
                If None, a new array is allocated and C above is taken to be zero
            beta: complex, default=0.0

            Returns
            -------
            out : NDarray
            """

        ...

else:
    ZGEMMWrapper = "ZGEMMWrapper"


@lru_cache(maxsize=None)
def get_zgemm(arr_type: type) -> ZGEMMWrapper:
    if arr_type is np.ndarray:
        from scipy.linalg.blas import zgemm

        def zgemm_sp(
            a: NDArray,
            b: NDArray,
            trans_a: Literal[0, 1, 2] = 0,
            trans_b: Literal[0, 1, 2] = 0,
            alpha: complex = 1.0,
            out: NDArray | None = None,
            beta: complex = 0.0,
        ):
            overwrite_c = 0
            out_ = None
            if out is not None:
                overwrite_c = 1
                out_ = np.asarray(out, "F")
            out_ = zgemm(
                alpha=alpha,
                a=a,
                b=b,
                c=out_,
                trans_a=trans_a,
                trans_b=trans_b,
                overwrite_c=overwrite_c,
                beta=beta,
            )
            if out is None:
                return out_
            if out_ is not out:
                out[:] = out_
            return out

        return zgemm_sp
    elif CUPY_INSTALLED:
        import cupy as cp

        if arr_type is cp.ndarray:
            from cupy.cublas import gemm

            def zgemm_cp(
                a: NDArray,
                b: NDArray,
                trans_a: Literal[0, 1, 2] = 0,
                trans_b: Literal[0, 1, 2] = 0,
                alpha: complex = 1.0,
                out: NDArray | None = None,
                beta: complex = 0.0,
            ):
                return gemm(trans_a, trans_b, a, b, out, alpha, beta)

            return zgemm_cp
    else:
        raise NotImplementedError(
            f"zgemm wrapper not implemented for array type '{arr_type}'"
        )


# def _check_zgemm_args(
#     a: NDArray, b: NDArray, trans_a: Literal[0, 1, 2],
#     trans_b: Literal[0, 1, 2], alpha: Complex, out: NDArray, beta: Complex
# ):
#     if not isinstance(a, NDArray):
#         raise TypeError("'a' must be an array instance. "
#                         f"got '{type(a)}'. ")
#     if a.ndim != 2 or not a.flags['C_CONTIGUOUS']:
#         raise ValueError("'a' must be a 2D Array with C-ordering. "
#                          f"got a.ndim = {a.ndim}, "
#                          f"a.flags['C_CONTIGUOUS'] = {a.flags['C_CONTIGUOUS']}")
#     if not isinstance(b, NDArray):
#         raise TypeError("'b' must be an array instance. "
#                         f"got '{type(b)}'. ")
#     if b.ndim != 2 or not b.flags['C_CONTIGUOUS']:
#         raise ValueError("'b' must be a 2D Array with C-ordering. "
#                          f"got b.ndim = {b.ndim}, "
#                          f"b.flags['C_CONTIGUOUS'] = {b.flags['C_CONTIGUOUS']}")
#     if trans_a not in [0, 1, 2]:
#         raise ValueError("'trans_a' must be either 0(for 'N'), 1(for 'T'), "
#                          f"or 2(for'C'). got {trans_a} (type '{type(trans_a)}').")
#     if trans_b not in [0, 1, 2]:
#         raise ValueError("'trans_b' must be either 0(for 'N'), 1(for 'T'), "
#                          f"or 2(for'C'). got {trans_b} (type '{type(trans_b)}').")
#     if not isinstance(alpha, Complex):
#         raise TypeError("'alpha' must be a complex number. "
#                         f"got {alpha} (type {type(alpha)}).")
#     if not isinstance(beta, Complex):
#         raise TypeError("'beta' must be a complex number. "
#                         f"got {beta} (type {type(beta)}).")
#     if out is None:
#         return
#     if not isinstance(out, NDArray):
#         raise TypeError("'out' must be an array instance. "
#                         f"got '{type(out)}'. ")
#     if out.ndim != 2 or not out.flags['C_CONTIGUOUS']:
#         raise ValueError("'out' must be a 2D Array with C-ordering. "
#                          f"got out.ndim = {out.ndim}, "
#                          f"out.flags['C_CONTIGUOUS'] = {out.flags['C_CONTIGUOUS']}")
