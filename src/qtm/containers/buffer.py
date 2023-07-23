# from __future__ import annotations
from typing import Self, Literal, Sequence, Union
from qtm.config import NDArray
__all__ = ['Buffer']

from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from qtm.gspace import GSpaceBase
from qtm.logger import warn


class Buffer(NDArrayOperatorsMixin, ABC):
    """QuantumMASALA's base container class.

    This abstract class implements an object that wraps a multidimensional
    array whose last dimension has a predefined length. The data in the
    array represents a (nested) sequence of vectors where the last dimension
    corresponds to the size of the 'basis' of the corresponding vector space.

    This class implements the following array-like behaviors:
    1. Indexing and assigning values to indexed arrays
    2. Indexing and assigning across the basis dimension via `BufferView`
    3. Implementing all NumPy ufuncs which includes most algebraic operation
       such as negation, addition, multiplication, comparision, etc.

    Parameters
    ----------
    gspc : GSpaceBase
        G-Space of the system
    data : NDArray
        Array to be wrapped. The length of its last dimension must equal the
        attribute `basis_size`.

    Notes
    -----
    Support for NumPy ufuncs are implemented using
    `numpy.lib.mixins.NDArrayOperatorsMixin`. Refer to:
    https://numpy.org/doc/stable/user/basics.dispatch.html

    .. warning:: Outputs of the ufuncs are cast back to the `Buffer` type
    (or its subclasses) if and only if the last axis matches the `basis_size`.
    This may lead to uncasted results if the ufunc involves transposition of
    axes. But, in the very rare case a buffer has multiple dimensions with the
    same length as basis_size, a warning is displayed to user indcating that
    the casting logic might be confused. This is **extremely** important when
    running in parallel as this class has no logic for performing reduction
    operations across data across the last index (which is distributed across
    processes).

    """

    @classmethod
    @abstractmethod
    def _get_basis_size(cls, gspc: GSpaceBase) -> int:
        """This is where the size of the basis is defined in classes
        implementing this container"""
        pass

    def __init__(self, gspc: GSpaceBase, data: NDArray):
        if not isinstance(gspc, GSpaceBase):
            raise TypeError("'gspc' must be a 'GSpaceBase' instance. "
                            f"got type {type(gspc)}")
        self._check_data(gspc, data, suppress_exc=False)

        self.gspc = gspc
        self._basis_size = self._get_basis_size(gspc)
        self._data = data

    @property
    def basis_size(self) -> int:
        """Size of the basis, which fixes the length of the last
        dimension"""
        return self._basis_size

    @property
    @abstractmethod
    def basis_type(self) -> Literal['r', 'g']:
        """Type of the basis. ``'r'`` corresponds to the real-space whereas
        ``'g'`` corresponds to the G-space defined by attribute `gspc`"""
        pass

    @classmethod
    def _check_data(cls, gspc: GSpaceBase,
                    data: np.ndarray, suppress_exc: bool = True):
        """Checks if the input `data` array has correct dtype, is contiguous in
        last dimension and has its length matching `basis_size`."""
        gspc.check_buffer(data)

        basis_size = cls._get_basis_size(gspc)
        if data.shape[-1] != basis_size:
            if suppress_exc:
                return False
            raise ValueError("'shape[-1]' of 'data' must be equal to 'basis_size'. "
                             f"got data.shape[-1] = {data.shape[-1]} "
                             f"basis_size = {basis_size}")

        if data.dtype != 'c16':
            if suppress_exc:
                return False
            raise ValueError("dtype of 'data' must be 'c16'. "
                             f"got data.dtype = {data.dtype}")

        # if data.strides[-1] != data.dtype.itemsize:
        #     if suppress_exc:
        #         return False
        #     raise ValueError("'data' must be atleast contiguous at the last axis. "
        #                      f"got data.strides[-1] = {data.strides[-1]} "
        #                      f"(expected {data.dtype.itemsize})")

        return True

    @property
    def data(self) -> NDArray:
        """Array containing the instance data. ``shape[-1]`` of this array equals
        `basis_size`"""
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array excluding the last dimension"""
        return self._data.shape[:-1]

    @property
    def rank(self) -> int:
        """Rank of the array, which equals ``data.ndim - 1``"""
        return self._data.ndim - 1

    class BufferView(NDArrayOperatorsMixin):
        """Class that provides interface to index the array `data`'s last
        dimension, allowing to slice by the last axis without determining the
        full shape of the array.

        Examples
        --------
        .. code-block::

            buf = Buffer.empty(gspc, (2, 5))
            buf.g[4] = 10  # Equivalent to buf.data[:, :, 4] = 10
            idx = [1, 2, 5]  # A boolean mask with length buf.basis_size also works
            print(buf.g[idx])  # Equivalent to buf.data[:, :, [1, 2, 5]]


        """

        def __init__(self, buffer):
            self.buffer: Buffer = buffer
            self.data: NDArray = self.buffer.data

            self.basis_size: int = self.buffer.basis_size
            self.shape: tuple[int, ...] = self.buffer.data.shape
            self.rank: int = self.buffer.rank

        def __getitem__(self, item) -> NDArray:
            if isinstance(item, tuple):
                raise TypeError(
                    "multidimensional indexing for 'BufferView' is disabled. "
                    "index is applied only to the last index.")
            return self.data[..., item]

        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                raise TypeError(
                    "multidimensional indexing for 'BufferView' is disabled. "
                    "index is applied only to the last index.")
            self.data[..., key] = value

        def __array__(self, dtype=None):
            if dtype is not None:
                return NotImplementedError(
                    "argument 'dtype' is not supported in 'BufferView.__array__. "
                    f"got {dtype}.")
            return self.data

    @property
    def r(self) -> BufferView:
        """If `basis_type` is ``'r'``, returns a ``BufferView`` instance."""
        if self.basis_type != 'r':
            raise AttributeError("'Buffer' property 'r' is undefined when "
                                 "'basis_type' is not 'r'. "
                                 f"got basis_type = {self.basis_type}")
        return self.BufferView(self)

    @property
    def g(self) -> BufferView:
        """If `basis_type` is ``'g'``, returns a ``BufferView`` instance."""
        if self.basis_type != 'g':
            raise AttributeError("'Buffer' property 'g' is undefined when "
                                 "'basis_type' is not 'g'. "
                                 f"got basis_type = {self.basis_type}")
        return self.BufferView(self)

    @classmethod
    def empty(cls, gspc: GSpaceBase, shape: Union[int, tuple[int, ...]]):
        """Creates a buffer with an empty array of given shape (with the basis
        dimension appended it)"""
        if not isinstance(gspc, GSpaceBase):
            raise TypeError("'gspc' must be a 'GSpaceBase' instance. "
                            f"got type {type(gspc)}")
        if isinstance(shape, int):
            shape = (shape, )
        basis_size = cls._get_basis_size(gspc)
        data = gspc.create_buffer((*shape, basis_size))
        return cls(gspc, data)

    @classmethod
    def zeros(cls, gspc: GSpaceBase, shape: Union[int, tuple[int, ...]]):
        out = cls.empty(gspc, shape)
        out.data[:] = 0
        return out

    def copy(self) -> Self:
        """Makes a copy of itself"""
        data = self._data.copy(order='C')
        return self.__class__(self.gspc, data)

    def reshape(self, shape: Sequence[int]) -> Self:
        """Similar implementation to reshaping multidimensional array"""
        try:
            data = self._data.reshape((*shape, self.basis_size))
            return self.__class__(self.gspc, data)
        except Exception as e:
            raise Exception("reshape failed. refer to below exception: ") from e

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
        if not isinstance(other, Buffer):
            raise TypeError("assignment and binary operations defined only between "
                            "instances of 'Buffer' and their subclasses. "
                            f"got {type(self)} and {type(other)}")
        if self.gspc != other.gspc:
            raise ValueError("'gspc' mismatch between 'Field' instances")
        if self.basis_size != other.basis_size:
            raise ValueError("'basis_size' mismatch between instances: "
                             f"got {self.basis_size} and {other.basis_size}")

    def __len__(self):
        if self.rank == 0:
            return 1
        return self.shape[0]

    def __iter__(self):
        if self.rank == 0:
            def generator(obj):
                yield obj
        else:
            def generator(obj):
                for idx in range(len(obj)):
                    yield obj[idx]
        return generator(self)

    def __getitem__(self, item) -> Self:
        self._check_slice(item)
        try:
            data = self._data[item]
            return self.__class__(self.gspc, data)
        except Exception as e:
            raise Exception("failed to slice field. refer to above exception "
                            "for further info.") from e

    def __setitem__(self, key, value: Self):
        self._check_slice(key)
        self._check_other(value)
        self._data[key] = value.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # print('debug1:', ufunc, 'method:', method)
        # print('inputs: ', inputs)
        # print('kwargs: ', kwargs)

        def cast_inout(args, add_newaxis: bool = True):
            if not isinstance(args, tuple):
                args = (args, )
            args_ndarray = []
            for arg in args:
                if isinstance(arg, Buffer):
                    if arg.gspc != self.gspc:
                        raise TypeError(
                            "mismatch in 'gspc' between two 'Buffer' instances."
                        )
                    if arg.basis_type != self.basis_type:
                        raise TypeError(
                            "mismatch in 'basis_type' between two 'Buffer' instances."
                        )
                    if arg.basis_size != self.basis_size:
                        raise TypeError(
                            "mismatch in 'basis_size' between two 'Buffer' instances."
                        )
                    args_ndarray.append(arg.data)
                else:
                    if add_newaxis:
                        args_ndarray.append(np.asarray(arg)[...: np.newaxis])
                    else:
                        args_ndarray.append(np.asarray(arg))
            return tuple(args_ndarray)

        inputs = cast_inout(inputs, add_newaxis=True)

        for inp in inputs:
            if self.basis_size in inp.shape[:-1]:
                warn(
                    "outputs of ufunc are cast to 'Buffer' instances when the last"
                    "dimension of the output array matches the size of the basis. "
                    "In the unlikely chance the input array has any dimension with the "
                    "same size that is not the last dimension, this warning pops up to "
                    "highlight a possible place for errors, especially when running "
                    "in parallel."
                    "Why do need such a large array/small basis at the first place?"
                    )

        out = kwargs.get('out', None)
        if out is not None:
            kwargs['out'] = cast_inout(kwargs['out'], add_newaxis=False)

        ufunc_out = getattr(ufunc, method)(*inputs, **kwargs)

        if out is not None:
            return out[0] if ufunc.nout == 1 else out

        if self._check_data(self.gspc, ufunc_out):
            ufunc_out = self.__class__(self.gspc, ufunc_out)
        return ufunc_out

    HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        if func not in self.HANDLED_FUNCTIONS:
            return NotImplemented
        func, supported_kwargs = self.HANDLED_FUNCTIONS[func]
        if any(kwarg not in supported_kwargs for kwarg in kwargs):
            raise NotImplementedError(
                "implementation only supports the follwing kwargs: "
                f"{str(supported_kwargs)[1:-1]}"
            )
        return func(*args, **kwargs)

    @classmethod
    def implements(cls, np_function, supported_kwargs):
        def decorator(func):
            cls.HANDLED_FUNCTIONS[np_function] = (func, supported_kwargs)
            return func
        return decorator


@Buffer.implements(np.sum, ('axis',))
def np_sum(arr: Buffer, axis=None):
    if axis is None:
        axis = tuple(range(arr.rank + 1))

    if isinstance(axis, int):
        axis = (axis, )

    axis = tuple(axis)

    reduce = False
    if arr.rank in axis or -1 in axis:
        reduce = True
    out = np.sum(arr.data, axis)

    if not reduce:
        return arr.__class__(arr.gspc, out)
    else:
        return out
