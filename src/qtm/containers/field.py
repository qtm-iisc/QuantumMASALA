"""Contains definitions of containers for storing periodic fields.

The implementation involves subclassing `BufferType` and specifying the G-Space
using a `GSpace` instance. This is done in the followoing way:

>>> grho: GSpace
... class FieldG_grho(FieldGType, gspc=grho):
...     pass
...
... rho_g = FieldG_grho.empty(2)

Convenience Functions `get_FieldG` and `get_FieldR` are defined to generate
the `FieldGType` and `FieldRType` instances respectively. So, the same can be
achieved in a single line:
>>> rho_g = get_FieldG(grho).empty(2)
... rho_r = get_FieldR(grho).empty(2)

Note that `get_FieldG` and `get_FieldR` are cached functions, so the same
`BufferType` class is returned for a given `GSpace` instance, allowing simpler
checking in array operations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type
__all__ = ["FieldGType", "get_FieldG", "FieldRType", "get_FieldR", "FieldType"]

from functools import lru_cache
from numbers import Number
from typing import Union
import numpy as np

from qtm.gspace import GSpace
from .buffer import BufferType
from qtm.config import MPI4PY_INSTALLED

from qtm.config import NDArray
from qtm.msg_format import *


class FieldGType(BufferType):
    """Container Template for storing periodic fields represented in G-Space
    (as fourier components)

    """

    gspc: GSpace = None
    basis_type = "g"
    basis_size: int = None
    ndarray: type = None

    def __init_subclass__(cls, gspc: GSpace):
        if not isinstance(gspc, GSpace):
            raise TypeError(type_mismatch_msg("gspc", gspc, GSpace))
        cls.gspc, cls.basis_size = gspc, gspc.size_g
        cls.ndarray = type(cls.gspc.g_cryst)

    def to_r(self) -> FieldRType:
        field_r = get_FieldR(self.gspc).empty(self.shape, "c16")
        self.gspc._g2r(self._data, field_r._data)
        return field_r

    def to_g(self) -> FieldGType:
        return self

    @property
    def data_g0(self) -> NDArray:
        return self.data[..., 0]


class FieldRType(BufferType):
    """Container Template for storing periodic fields represented in real space"""

    gspc: GSpace = None
    basis_type = "r"
    basis_size: int = None
    ndarray: type = None

    def __init_subclass__(cls, gspc: GSpace):
        if not isinstance(gspc, GSpace):
            raise TypeError(type_mismatch_msg("gspc", gspc, GSpace))
        cls.gspc, cls.basis_size = gspc, gspc.size_r
        cls.ndarray = type(cls.gspc.g_cryst)

    def to_g(self) -> FieldGType:
        field_g = get_FieldG(self.gspc).empty(self.shape, "c16")
        self.gspc._r2g(self._data, field_g._data)
        return field_g

    def to_r(self) -> FieldRType:
        return self

    def integrate_unitcell(self, other=None, axis=-1) -> NDArray | Number:
        # TODO: Update docstrings
        """Evaluates the integral of the field across the unit cell.

        Effectively a `numpy.sum` operation involving the last axis + input
        axes, follwed by its product with the differential volume
        `qtm.gspace.GSpaceBase.reallat_dv`

        Returns
        -------
        NDArray | Number
            `self.data` summed across the last axis.
        """
        if other is not None:
            return (
                np.sum(np.sum(self._data * other, axis=-1), axis=axis)
                * self.gspc.reallat_dv
            )
        else:
            return np.sum(self, axis=axis) * self.gspc.reallat_dv

FieldType = Union[FieldGType, FieldRType]

@lru_cache(maxsize=None)
def get_FieldG(gspc: GSpace) -> Type[FieldGType]:
    if MPI4PY_INSTALLED:
        from qtm.mpi.gspace import DistGSpace
        from qtm.mpi.containers import get_DistFieldG

        if isinstance(gspc, DistGSpace):
            return get_DistFieldG(gspc)

    class FieldG(FieldGType, gspc=gspc):
        pass

    return FieldG


@lru_cache(maxsize=None)
def get_FieldR(gspc: GSpace) -> Type[FieldRType]:
    if MPI4PY_INSTALLED:
        from qtm.mpi.gspace import DistGSpace
        from qtm.mpi.containers import get_DistFieldR

        if isinstance(gspc, DistGSpace):
            return get_DistFieldR(gspc)

    class FieldR(FieldRType, gspc=gspc):
        pass

    return FieldR
