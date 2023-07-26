# from __future__ import annotations
from typing import Self, Literal, Sequence, Union, Optional
from qtm.config import NDArray
__all__ = ['Buffer']

from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from qtm.gspace import GSpaceBase


class Buffer(NDArrayOperatorsMixin, ABC):
    """QuantumMASALA's base container class.

    This abstract class implements an object that wraps a multidimensional
    array whose last dimension has a predefined length. The data in the
    array represents a (nested) sequence of vectors where the last dimension
    corresponds to the size of the 'basis' of the corresponding vector space.

    This class implements the following array-like behaviors:
    1. Indexing and assigning values to indexed arrays
    2. Indexing and assigning across the basis dimension via `BufferView`
    3. Implementing NumPy ufuncs which includes algebraic operation
       such as negation, addition, multiplication, comparision, etc.
       and scalar functions like exp, sin, cos, tan, etc.

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

    This class limits the list of supported NumPy Ufuncs to only:
    1. Scalar Ufuncs that is applied to every element of the (broadcasted)
    input array arguments
    2. Ufunc's 'reduce' method, which is applicable to only scalar
    binary operations.

    When the basis is distibuted across processes, the corresponding
    'DistBuffer' class will handle reductions across the last dimension,
    which is now parallelized and thus requires an MPI_ALLreduce operation.
    """

    @classmethod
    @abstractmethod
    def _get_basis_size(cls, gspc: GSpaceBase) -> int:
        """This is where the size of the basis is defined in classes
        implementing this container"""
        pass

    def __init__(self, gspc: GSpaceBase, data: NDArray):
        if not isinstance(gspc, GSpaceBase):
            raise TypeError(f"'gspc' must be a '{GSpaceBase}' instance. "
                            f"got '{type(gspc)}'.")
        self._check_data(gspc, data, suppress_exc=False)

        self.gspc = gspc
        self._basis_size = self._get_basis_size(gspc)
        if not isinstance(data, NDArray):
            raise TypeError("'data' must be an array instance. "
                            f"got '{type(data)}'.")
        if data.shape[-1] != self._basis_size:
            raise ValueError("'")
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
                    "index is applied only to the last dimension.")
            return self.data[..., item]

        def __setitem__(self, key, value):
            if isinstance(key, tuple):
                raise TypeError(
                    "multidimensional indexing for 'BufferView' is disabled. "
                    "index is applied only to the last dimension.")
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
            raise TypeError(f"'gspc' must be a '{GSpaceBase}' instance. "
                            f"got '{type(gspc)}'")
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

    @abstractmethod
    def to_r(self) -> Self:
        pass

    @abstractmethod
    def to_g(self) -> Self:
        pass

    def reshape(self, shape: Sequence[int]) -> Self:
        """Similar implementation to reshaping multidimensional array"""
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
        if not isinstance(other, Buffer):
            raise TypeError("assignment and binary operations defined only between "
                            "instances of 'Buffer' and their subclasses. "
                            f"got {type(self)} and {type(other)}")
        if self.gspc != other.gspc:
            raise ValueError("'gspc' mismatch between 'Buffer' instances")
        if self.basis_size != other.basis_size:
            raise ValueError("'basis_size' mismatch between 'Buffer' instances: "
                             f"got {self.basis_size} and {other.basis_size}")

    def __getitem__(self, item) -> Self:
        self._check_slice(item)
        try:
            data = self._data[item]
            return self.__class__(self.gspc, data)
        except Exception as e:
            raise Exception("failed to slice field. refer to the rest of the "
                            "exception message for further info.") from e

    def __setitem__(self, key, value: Self):
        self._check_slice(key)
        self._check_other(value)
        self._data[key] = value.data

    def __len__(self):
        if self.rank == 0:
            raise TypeError("'Buffer' instance is scalar (rank=0).")
        return self.shape[0]

    def __iter__(self):
        if self.rank == 0:
            raise TypeError("'Buffer' instance is scalar (rank=0).")
        else:
            def generator(obj):
                for idx in range(len(obj)):
                    yield obj[idx]
        return generator(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        cast_out_to_buffer = True  # If True, the output will be converted to a 'Buffer' instance
        if method == 'reduce':
            axis = kwargs.get('axis', None)
            ndim = self._data.ndim
            if axis is None:
                axis = tuple(idim for idim in range(ndim))
            elif not isinstance(axis, tuple):
                axis = (axis, )
            # If last axis representing the basis is reduced, don't cast output to buffer
            if ndim - 1 in axis or -1 in axis:
                cast_out_to_buffer = False
        elif method == '__call__':
            if ufunc.signature is not None:
                if ufunc is not np.matmul:
                    raise NotImplementedError(
                        "Only scalar ufuncs are supported."
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
                if isinstance(inp, Buffer):
                    if inp.gspc != self.gspc:
                        raise TypeError(
                            "mismatch in 'gspc' between two 'Buffer' instances."
                        )
                    if inp.basis_type != self.basis_type:
                        raise TypeError(
                            "mismatch in 'basis_type' between two 'Buffer' instances."
                        )
                    if inp.basis_size != self.basis_size:
                        raise TypeError(
                            "mismatch in 'basis_size' between two 'Buffer' instances."
                        )
                    ufunc_inp.append(inp.data)
            else:
                ufunc_inp.append(np.asarray(inp)[..., np.newaxis])

        # If kwarg 'out' is given and if any one is a 'Buffer' instance, the
        # data array is extracted. Unlike in input, no casting is done here
        outputs = kwargs.get('out', ())
        ufunc_out = []
        if outputs:
            for out in outputs:
                if isinstance(out, Buffer):
                    if out.gspc != self.gspc:
                        raise TypeError(
                            "mismatch in 'gspc' between two 'Buffer' instances."
                        )
                    if out.basis_type != self.basis_type:
                        raise TypeError(
                            "mismatch in 'basis_type' between two 'Buffer' instances."
                        )
                    if out.basis_size != self.basis_size:
                        raise TypeError(
                            "mismatch in 'basis_size' between two 'Buffer' instances."
                        )
                    ufunc_out.append(out.data)
                else:
                    ufunc_out.append(out)
            kwargs['out'] = tuple(ufunc_out)

        ufunc_out = getattr(ufunc, method)(*ufunc_inp, **kwargs)
        # If 'cast_out_to_buffer' is True, then they are cast to the same type
        # as self, else they are returned as bare arrays.
        if isinstance(ufunc_out, tuple):
            if cast_out_to_buffer:
                return tuple(type(self)(self.gspc, out) for out in ufunc_out)
            else:
                return ufunc_out
        elif cast_out_to_buffer:
            return type(self)(self.gspc, ufunc_out)
        else:
            return ufunc_out
