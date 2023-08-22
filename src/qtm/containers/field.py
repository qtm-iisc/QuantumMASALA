from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Literal
__all__ = ['Field', 'FieldG', 'FieldR']

from abc import ABC, abstractmethod
import numpy as np

from qtm.gspace import GSpace
from .buffer import Buffer
from qtm.config import qtmconfig

from qtm.config import NDArray
from qtm.msg_format import *


class Field(Buffer, ABC):
    """Represents Scalar/Vector/Tensor fields within the unit cell of crystal.

    Requires `gspc` to strictly be a `qtm.gspace.GSpace` instance as the
    set of G-vectors in G-Space must map to themselves under any of the
    crystal's symmetry operations. Constructing a G-space with a cutoff
    sphere that is within the FFT Grid Box will result in such sets.
    """

    gspc: GSpace

    @abstractmethod
    def __new__(cls, gspc: GSpace, data: NDArray):
        return Buffer.__new__(cls, gspc, data)

    def __init__(self, gspc: GSpace, data: NDArray):
        if not isinstance(gspc, GSpace):
            raise TypeError(type_mismatch_msg('gspc', gspc, GSpace))
        Buffer.__init__(self, gspc, data)


class FieldG(Field):
    """`qtm.containers.Field` subclass implementing the G-Space representation
    of a scalar/vector/tensor field.
    """
    def __new__(cls, gspc: GSpace, data: NDArray):
        if not qtmconfig.mpi4py_installed:
            return Field.__new__(cls, gspc, data)

        from qtm.mpi import DistGSpace, DistFieldG, DistBuffer
        if isinstance(gspc, DistGSpace):
            return DistBuffer.__new__(DistFieldG, gspc, data)
        return Field.__new__(cls, gspc, data)

    @classmethod
    def _get_basis_size(cls, gspc: GSpace):
        return gspc.size_g

    @property
    def basis_type(self) -> Literal['g']:
        return 'g'

    def to_r(self) -> FieldR:
        gspc = self.gspc
        data = gspc.g2r(self.data)
        return FieldR(gspc, data)

    def to_g(self) -> FieldG:
        return self


class FieldR(Field):
    """`qtm.containers.Field` subclass implementing the real-space
    representation of a scalar/vector/tensor field.
    """

    def __new__(cls, gspc: GSpace, data: NDArray):
        if not qtmconfig.mpi4py_installed:
            return Buffer.__new__(FieldR, gspc, data)

        from qtm.mpi import DistGSpace, DistFieldR, DistBuffer
        if isinstance(gspc, DistGSpace):
            return DistBuffer.__new__(DistFieldR, gspc, data)
        return Buffer.__new__(FieldR, gspc, data)

    @classmethod
    def _get_basis_size(cls, gspc: GSpace):
        return gspc.size_r

    @property
    def basis_type(self) -> Literal['r']:
        return 'r'

    def to_r(self) -> FieldR:
        return self

    def to_g(self) -> FieldG:
        gspc = self.gspc
        data = gspc.r2g(self.data)
        return FieldG(gspc, data)

    def integrate_unitcell(self) -> NDArray | complex:
        #TODO: Update docstrings
        """Evaluates the integral of the field across the unit cell.

        Effectively a `numpy.sum` operation involving the last axis + input
        axes, follwed by its product with the differential volume
        `qtm.gspace.GSpaceBase.reallat_dv`

        Parameters
        ----------
        axis : int | Sequence[int] | None, default=None
            Axes of the multidimensional field to integrate. By default,
            all axes are summed up.
        Returns
        -------
        NDArray | complex

        """
        return np.sum(self, axis=-1) * self.gspc.reallat_dv
