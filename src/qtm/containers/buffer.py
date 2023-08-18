from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Sequence
__all__ = ['Buffer']

from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from qtm.gspace import GSpaceBase

from qtm.config import NDArray
from qtm.msg_format import *


class Buffer(NDArrayOperatorsMixin, ABC):
    """QuantumMASALA's base container class.

    This abstract class wraps a multidimensional array whose last dimension
    has a fixed length corresponding to the basis of the space. For
    real-space, it corresponds to the number of points in the grid defined
    within the unit cell and for G-space, it corresponds to the number of
    G-vectors. It contains array creation routines (`empty`, `zeros`, `copy`,
    `from_array`, etc.) and supports indexing, slicing and scalar NumPy
    Universal functions, enabling algebraic and logical operators
    like ``+``, ``-``, ``*``, ``/``, ``>``, ``<``, etc.

    Parameters
    ----------
    gspc : GSpaceBase
        G-Space of the system, defines the function's basis.
    data : NDArray
        Array to be wrapped. The length of its last dimension must equal the
        attribute `basis_size`.

    Notes
    -----
    Support for NumPy ufuncs are implemented using
    `numpy.lib.mixins.NDArrayOperatorsMixin`. Refer to:
    https://numpy.org/doc/stable/user/basics.dispatch.html

    This class limits the list of supported NumPy Ufuncs to only:
    1. Scalar Ufuncs that is applied to every element of the (broadcasted)
    input array arguments
    2. NumPy's matmul
    3. NumPy Ufunc's 'reduce' method, which is applicable to only scalar
    binary operations.
    Note that when any reduce operations is applied to the last axis, the
    resulting output will be an array instead of a `Buffer` instance.

    Notes about `qtm.mpi.containers.DistBuffer`
    -------------------------------------------
    When `Buffer` subclasses are initialized with a
    `qtm.mpi.gspace.DistGSpaceBase` instance representing the G-Space basis
    distributed across multiple processes, the analogous distributed
    container `qtm.mpi.containers.DistBuffer` is instead created and returned.
    Refer to its documentation for further info about the distributed
    container and its implementation.
    """

    @classmethod
    @abstractmethod
    def _get_basis_size(cls, gspc: GSpaceBase) -> int:
        """Defines attributes `basis_size` in subclasses"""
        pass

    @abstractmethod
    def __new__(cls, gspc: GSpaceBase, data: NDArray):
        """This method is overloaded to allow the serial class to instantiate
        the corresponding distributed version when the input `gspc` is a
        `qtm.mpi.gspace.DistGSpaceBase` instance.
        """
        return super().__new__(cls)

    def __init__(self, gspc: GSpaceBase, data: NDArray):
        if not isinstance(gspc, GSpaceBase):
            raise TypeError(type_mismatch_msg('gspc', gspc, GSpaceBase))
        self._check_data(gspc, data, suppress_exc=False)

        self.gspc: GSpaceBase = gspc
        """`GSpaceBase` instance defining both real-space and G-Space basis"""

        self._basis_size: int = self._get_basis_size(gspc)
        """Size of the basis; for real-space it is the number of mesh points 
        whereas for G-Space, it is the number of G-vectors within cutoff"""

        if not isinstance(data, NDArray):
            raise TypeError(type_mismatch_msg(
                'data', data, "a supported 'ndarray' instance"
            ))
        self._data: NDArray = data
        """Array instance containing the data"""

    @property
    def basis_size(self) -> int:
        """Size of the basis, which fixes the length of the last
        dimension"""
        return self._basis_size

    @property
    @abstractmethod
    def basis_type(self) -> Literal['r', 'g']:
        """Type of the basis. ``'r'`` corresponds to the real-space whereas
        ``'g'`` corresponds to the G-space defined by `gspc`"""
        pass

    @classmethod
    def _check_data(cls, gspc: GSpaceBase,
                    data: NDArray, suppress_exc: bool = True):
        """Checks if the input `data` array is of the correct type, has dtype
        ``'c16'`` and has its length along the last axis equal to
        `basis_size`."""
        gspc.check_array_type(data)

        basis_size = cls._get_basis_size(gspc)
        if data.dtype != 'c16':
            if suppress_exc:
                return False
            raise ValueError(value_mismatch_msg(
                'data.dtype', data.dtype, 'c16'
            ))

        if data.ndim == 0:
            if suppress_exc:
                return False
            raise ValueError(value_mismatch_msg(
                'data.ndim', data.ndim, 'a positive integer'
            ))
        if data.shape[-1] != basis_size:
            if suppress_exc:
                return False
            raise ValueError(value_mismatch_msg(
                'data.shape[-1]', data.shape[-1], f"'basis_size'={basis_size}"
            ))
        return True

    @property
    def data(self) -> NDArray:
        """Array containing the data of the periodic function"""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array excluding the last dimension"""
        return self._data.shape[:-1]

    @property
    def rank(self) -> int:
        """Rank of the array, which equals ``data.ndim - 1``"""
        return self._data.ndim - 1

    @classmethod
    def empty(cls, gspc: GSpaceBase,
              shape: int | Sequence[int] | None) -> Buffer:
        """Creates an empty `Buffer` instance of given shape.

        Parameters
        ----------
        gspc : GSpaceBase
            G-Space of the system
        shape : int | Sequence[int] | None
            Shape of the empty `Buffer`. If None, results in a scalar `Buffer`.
        """
        if not isinstance(gspc, GSpaceBase):
            raise TypeError(type_mismatch_msg('gspc', gspc, GSpaceBase))
        if shape is None:
            shape = ()
        elif isinstance(shape, int):
            shape = (shape, )
        basis_size = cls._get_basis_size(gspc)
        data = gspc.allocate_array((*shape, basis_size))
        return cls(gspc, data)

    @classmethod
    def zeros(cls, gspc: GSpaceBase,
              shape: int | Sequence[int] | None):
        """Creates an `Buffer` instance of given shape containing zeros.

        Parameters
        ----------
        gspc : GSpaceBase
            G-Space of the system
        shape : int | Sequence[int] | None
            Shape of the empty `Buffer`. If None, results in a scalar `Buffer`.
        """
        out = cls.empty(gspc, shape)
        out.data[:] = 0
        return out

    @classmethod
    def from_array(cls, gspc: GSpaceBase, data: NDArray):
        """Creates an empty `Buffer` instance from input array

        Parameters
        ----------
        gspc : GSpaceBase
            G-Space of the system
        data : NDArray
            Data array to be cast into a `Buffer`. The array is copied.
        """
        return cls(gspc, data.astype('c16').copy(order='C'))

    def copy(self) -> Buffer:
        """Makes a copy of itself"""
        data = self._data.copy(order='C')
        return self.__class__(self.gspc, data)

    def conj(self) -> Buffer:
        """Returns the complex conjugate of its data, cast to the same type
        as the instance."""
        return self.__class__(self.gspc, self._data.conj())

    @abstractmethod
    def to_r(self) -> Buffer:
        """If `basis_type` is ``'g'``, returns the Backwards Fourier Transform
        of the instances' data cast to its dual `Buffer` subtype.
        Else returns itself."""
        pass

    @abstractmethod
    def to_g(self) -> Buffer:
        """If `basis_type` is ``'r'``, returns the Forward Fourier Transform
        of the instances' data cast to its dual `Buffer` subtype.
        Else returns itself."""
        pass

    def reshape(self, shape: int | Sequence[int]) -> Buffer:
        """Returns a new instance whose data is reshaped according to input
        `shape`

        Parameters
        ----------
        shape : int | Sequence[int]
            Shape of the resulting `Buffer` instance

        Raises
        ------
        Exception
            Raised if reshape failed or does not yield an array whose last
            dimension matches `basis_size`.
        """
        if isinstance(shape, int):
            shape = (shape, )
        try:
            data = self._data.reshape((*shape, self.basis_size))
            return self.__class__(self.gspc, data)
        except Exception as e:
            raise Exception("reshape failed. refer to the rest of the exception "
                            "for further info.") from e

    def _check_slice(self, item):
        if self.rank == 0:
            raise IndexError("cannot be indexed as it is scalar (rank=0)")
        if isinstance(item, tuple):
            if len(item) > self.rank:
                raise IndexError(
                    f"too many indices: only {self.rank} dimensions while "
                    f"{len(item)} were indexed"
                )

    def _check_other(self, other):
        if type(self) is not type(other):
            raise TypeError(f"operations defined between {Buffer} instances "
                            f"must match the same subclass. "
                            f"got {type(self)} and {type(other)}.")
        if self.gspc is not other.gspc:
            raise ValueError(obj_mismatch_msg(
                'self.gspc', self.gspc, 'other.gspc', other.gspc
            ))

    def __getitem__(self, item) -> Buffer:
        self._check_slice(item)
        try:
            data = self._data[item]
            return self.__class__(self.gspc, data)
        except Exception as e:
            raise Exception("failed to slice field. refer to the rest of the "
                            "exception message for further info.") from e

    def __setitem__(self, key, value: Buffer):
        self._check_slice(key)
        self._check_other(value)
        self._data[key] = value.data

    def __len__(self):
        if self.rank == 0:
            raise TypeError(f"'{type(self)}' instance is scalar (rank=0).")
        return self.shape[0]

    def __iter__(self):
        if self.rank == 0:
            raise TypeError(f"'{type(self)}' instance is scalar (rank=0).")
        else:
            def generator(obj):
                for idx in range(len(obj)):
                    yield obj[idx]
        return generator(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method in ['__call__', 'reduce']:
            if ufunc.signature is not None and ufunc is not np.matmul:
                raise NotImplementedError(
                    "Only scalar ufuncs and matmul are supported."
                )
            if ufunc == np.matmul and 'axes' in kwargs:
                raise NotImplementedError(
                    "'axes' keyword for 'matmul' not supported"
                )
        else:
            raise NotImplementedError(
                "'Buffer' instances only support '__call__' and 'reduce' methods "
                "of NumPy Ufuncs. If needed, you can operate on their data directly "
                "by accessing its 'data' property, but it might result it "
                "unexpected behavior when running in parallel."
            )

        # Iterating through inputs and casting them to NDArrays
        ufunc_inp = []
        for inp in inputs:
            if isinstance(inp, Buffer):
                self._check_other(inp)
                ufunc_inp.append(inp.data)
            else:
                ufunc_inp.append(np.asarray(inp, like=self.data))

        # If kwarg 'out' is given and if any one is a 'Buffer' instance, the
        # data array is extracted. Unlike in input, no casting is done here
        outputs = kwargs.get('out', ())
        ufunc_out = []
        if outputs:
            for out in outputs:
                if isinstance(out, Buffer):
                    self._check_other(out)
                    ufunc_out.append(out.data)
                else:
                    ufunc_out.append(out)
            kwargs['out'] = tuple(ufunc_out)

        if method == '__call__':
            if 'order' in kwargs:
                if kwargs['order'] != 'C':
                    raise ValueError("'order' keyword only suports 'C'.")
            kwargs['order'] = 'C'

        ufunc_out = getattr(ufunc, method)(*ufunc_inp, **kwargs)
        # If 'cast_out_to_buffer' is True, then they are cast to the same type
        # as self, else they are returned as bare arrays.
        if isinstance(ufunc_out, tuple):
            return tuple(type(self)(self.gspc, out)
                         if out.ndim > 0 and out.shape[-1] == self.basis_size
                         else out for out in ufunc_out)
        elif ufunc_out.ndim > 0 and ufunc_out.shape[-1] == self.basis_size:
            return type(self)(self.gspc, ufunc_out)
        else:
            return ufunc_out
