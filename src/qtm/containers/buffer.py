from __future__ import annotations

import numbers
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Sequence, Self
__all__ = ['BufferType']

from abc import ABC, abstractmethod
from numbers import Number
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from qtm.gspace import GSpaceBase

from qtm.config import NDArray


class BufferType(NDArrayOperatorsMixin, ABC):
    """QuantumMASALA's base container class template.

    This abstract class wraps a multidimensional array whose last dimension
    has a fixed length corresponding to the basis of the space. For
    real-space, it corresponds to the number of points in the grid defined
    within the unit cell and for G-space, it corresponds to the number of
    G-vectors. It contains array creation routines (`empty`, `zeros`, `copy`,
    `from_array`, etc.) and supports indexing, slicing and scalar NumPy
    Universal functions, enabling algebraic and logical operators
    like ``+``, ``-``, ``*``, ``/``, ``>``, ``<``, etc.

    The class cannot be used as-is and requires subclassing because the
    `GSpaceBase` instance `gspc` describing the basis is defined to be a
    class attribute. Refer to `qtm.containers.field` submodule for an
    example on how to use this class.

    Parameters
    ----------
    data : NDArray
        Array to be wrapped. The length of its last dimension must be equal
        to the class attribute `basis_size`.

    Notes
    -----
    No checks are done to `data` in the `__init__` method. If you want to
    initialize from the existing data, please use the `from_array`
    class method for the purpose.

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
    """

    # Below are the class attributes that will be initialized by the subclass'
    # `__init_subclass__` method, which takes a `GSpaceBase` argument.
    gspc: GSpaceBase = None
    """`GSpaceBase` instance representing the basis of the space"""
    basis_type: Literal['r', 'g'] = None
    """If ``'r'``, data is represented in real-space basis and if ``'g'``, data
    is represented in G-Space"""
    basis_size: int = None
    """Size of the basis. The lenght of the last axis of the data array must
    be equal to this length"""
    ndarray: type = None
    """Array Type of `data`. Matches the type of the arrays in `gspc` 
    (`gspc.g_cryst`)"""

    @abstractmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    def __init__(self, data: NDArray):
        # self._check_data(data, suppress_exc=False)
        self._data: NDArray = data
        """Array instance containing the data"""

    @classmethod
    def _check_data(cls, data: NDArray, suppress_exc: bool = True):
        """Checks if the input `data` array is of the correct type, and has
        its length along the last axis equal to `basis_size`."""
        try:
            assert type(data) is cls.ndarray
            assert data.ndim > 0
            assert data.shape[-1] == cls.basis_size
        except AssertionError as e:
            if suppress_exc:
                return False
            raise ValueError(
                "input 'data' failed assertion checks. "
                "Refer to error messages above.") from e

    @property
    def data(self) -> NDArray:
        """(``(..., basis_size)``) Array containing the data of the
        periodic function"""
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
    def empty(cls, shape: int | Sequence[int] = (),
              dtype: str = 'c16') -> Self:
        """Creates a empty `BufferType` instance of given shape.

        Parameters
        ----------
        shape : int | Sequence[int], default=()
            Shape of the empty `BufferType` instance to create.
            For scalar/1D `BufferType`, `shape` must be `()`.
        dtype: str, default='c16'
            String representing the array dtype to be allocated.

        Returns
        -------
        A `BufferType` instance containing an empty data array of
        shape ``(*shape, basis_size)`` and dtype `dtype`.
        """
        if shape is None:
            shape = ()
        elif isinstance(shape, int):
            shape = (shape, )
        data = cls.gspc.allocate_array((*shape, cls.basis_size), dtype=dtype)
        return cls(data)

    @classmethod
    def zeros(cls, shape: int | Sequence[int] = (),
              dtype: str = 'c16') -> Self:
        """Creates a `BufferType` instance of given shape containing zeros.

        Parameters
        ----------
        shape : int | Sequence[int], default=()
            Shape of the empty `BufferType` instance to create.
            For scalar/1D `BufferType`, `shape` must be `()`.
        dtype: str, default='c16'
            String representing the array dtype to be allocated.

        Returns
        -------
        A `BufferType` instance containing a zero data array of
        shape ``(*shape, basis_size)`` and dtype `dtype`.
        """
        out = cls.empty(shape, dtype)
        out.data[:] = 0
        return out

    @classmethod
    def from_array(cls, data: NDArray) -> Self:
        """Creates a `BufferType` instance from values in `data` array.

        Parameters
        ----------
        data : NDArray
            Input data array. Must be an `ndarray` instance with shape
            ``(..., basis_size)``.

        Returns
        -------
        A `BufferType` instance containing a copy of the `data` array.
        """
        assert type(data) is cls.ndarray
        assert data.shape[-1] == cls.basis_size
        return cls(data.copy(order='C'))

    def copy(self) -> Self:
        """Makes a copy of itself"""
        data = self._data.copy(order='C')
        return self.__class__(data)

    def conj(self) -> Self:
        """Returns the complex conjugate of its data, cast to the same type
        as the instance."""
        return self.__class__(self._data.conj())

    @abstractmethod
    def to_r(self) -> Self:
        """If `basis_type` is ``'g'``, returns the Backwards Fourier Transform
        of the instances' data cast to its dual `BufferType` subtype.
        Else returns itself."""
        pass

    @abstractmethod
    def to_g(self) -> Self:
        """If `basis_type` is ``'r'``, returns the Forward Fourier Transform
        of the instances' data cast to its dual `BufferType` subtype.
        Else returns itself."""
        pass

    def reshape(self, shape: int | Sequence[int]) -> Self:
        """Returns a new instance whose data is reshaped according to input
        `shape`

        Parameters
        ----------
        shape : int | Sequence[int]
            Shape of the resulting `BufferType` instance

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
            return self.__class__(data)
        except Exception as e:
            raise Exception("reshape failed. refer to the rest of the exception "
                            "for further info.") from e

    def __getitem__(self, item) -> Self:
        if not isinstance(item, tuple):
            item = (item, )
        sl = (*item, Ellipsis, slice(None))
        try:
            data = self._data[sl]
            return self.__class__(data)
        except Exception as e:
            raise Exception("slicing failed. refer to the rest of the "
                            "exception message for further info.") from e

    def __setitem__(self, key, value: Self):
        try:
            view = self._data[key]
            assert view.shape[-1] == self.basis_size
        except Exception as e:
            raise Exception(
                "slicing failed. Refer to the rest of the exception message "
                "for further info. Make sure you are not slicing across the "
                "last axis. If so, aceess it via the 'data' attribute.") from e
        self._data[key] = value.data if type(self) is type(value) else value

    def __len__(self):
        if self.rank == 0:
            raise TypeError(f"'{type(self)}' instance is scalar (rank=0).")
        return self.shape[0]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method in ['__call__', 'reduce']:
            if ufunc.signature is not None and ufunc is not np.matmul:
                raise NotImplementedError(
                    "Only scalar ufuncs and 'matmul' are supported."
                )
            if ufunc == np.matmul and 'axes' in kwargs:
                raise NotImplementedError(
                    "'axes' keyword for 'matmul' not supported"
                )
        else:
            raise NotImplementedError(
                "'BufferType' instances only support '__call__' and 'reduce' methods "
                "of NumPy Ufuncs. If needed, you can operate on their data directly "
                "by accessing its 'data' property, but it might result it "
                "unexpected behavior when running in parallel."
            )

        # Iterating through inputs and casting them to NDArrays
        ufunc_inp = []
        for inp in inputs:
            if isinstance(inp, (Number, self.ndarray)):
                ufunc_inp.append(inp)
            elif isinstance(inp, type(self)):
                ufunc_inp.append(inp.data)
            else:
                return NotImplemented

        # If kwarg 'out' is given and if any one is a 'Buffer' instance, the
        # data array is extracted. Unlike in input, no casting is done here
        outputs = kwargs.get('out', ())
        if outputs:
            ufunc_out = []
            for out in outputs:
                if isinstance(out, (numbers.Number, self.ndarray)):
                    ufunc_out.append(out)
                elif isinstance(out, type(self)):
                    ufunc_out.append(out.data)
                else:
                    return NotImplemented
            kwargs['out'] = tuple(ufunc_out)

        if method == '__call__':
            if 'order' in kwargs:
                if kwargs['order'] != 'C':
                    raise ValueError("'order' keyword only suports 'C'.")
            # kwargs['order'] = 'C'

        ufunc_out = getattr(ufunc, method)(*ufunc_inp, **kwargs)
        # If 'cast_out_to_buffer' is True, then they are cast to the same type
        # as self, else they are returned as bare arrays.
        if outputs:
            return outputs[0] if len(outputs) == 1 else outputs
        elif isinstance(ufunc_out, tuple):
            return tuple(type(self)(out)
                         if out.ndim > 0 and out.shape[-1] == self.basis_size
                         else out for out in ufunc_out)
        elif ufunc_out.ndim > 0 and ufunc_out.shape[-1] == self.basis_size:
            return type(self)(ufunc_out)
        else:
            return ufunc_out
