from __future__ import annotations
__all__ = ['GField', 'RField']

from abc import ABC, abstractmethod
from typing import Union, Optional

import numpy as np

from quantum_masala.core import GSpace
from quantum_masala import config


def _check_shape(shape):
    if not isinstance(shape, tuple):
        raise TypeError("'shape' must be a tuple of positive integers. "
                        f"got {type(shape)}")
    for i, dim in enumerate(shape):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("'shape' must contain only positive integers. "
                             f"got {shape}")


class Field(ABC):

    @abstractmethod
    def __init__(self, gspc: GSpace, shape: Union[int, tuple[int, ...]],
                 data: Optional[np.ndarray] = None):
        self.pwcomm = config.pwcomm
        self.gspc: GSpace = gspc

        if isinstance(shape, int):
            shape = (shape, )
        _check_shape(shape)
        self.shape = shape
        self._data = None

    @property
    def data(self):
        return self._data

    @classmethod
    def empty(cls, gspc: GSpace, shape: Union[int, tuple[int, ...]]):
        return cls(gspc, shape)

    @classmethod
    def zeros(cls, gspc: GSpace, shape: Union[int, tuple[int, ...]]):
        out = cls(gspc, shape)
        out._data[:] = 0
        return out

    @classmethod
    def empty_like(cls, f: Field):
        if not isinstance(f, Field):
            raise ValueError(f"'f' must an instance of 'Field'. got {type(f)}")
        return cls.empty(f.gspc, f.shape)

    @classmethod
    def zeros_like(cls, f: Field):
        if not isinstance(f, Field):
            raise ValueError(f"'f' must an instance of 'Field'. got {type(f)}")
        return cls.zeros(f.gspc, f.shape)

    @classmethod
    @abstractmethod
    def from_array(cls, gspc: GSpace, arr: np.ndarray,
                   copy_arr: bool = True):
        pass

    def copy(self):
        return self.__class__.from_array(self.gspc, self._data, copy_arr=True)

    def Bcast(self):
        self.pwcomm.world_comm.Bcast(self._data)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__.from_array(self.gspc, -self.data, copy_arr=False)

    def _check_other(self, other, check_subclass=True, check_gspc=True):
        if not isinstance(other, Field):
            raise TypeError("operation defined only between 'Field' instances: "
                            f"got {type(other)}")
        if check_subclass and not isinstance(other, self.__class__):
            raise TypeError("operation can only be peformed between instannces with "
                            f"same 'Field' subtype. got {type(self)}, {type(other)}")
        if check_gspc and self.gspc != other.gspc:
            raise ValueError("'gspc' do not match between the operands.")

    @abstractmethod
    def _update_shape(self):
        pass

    def __add__(self, other):
        self._check_other(other)
        out_data = self._data + other.data
        return self.__class__.from_array(self.gspc, out_data, copy_arr=False)

    def __iadd__(self, other):
        self._check_other(other)
        self._data += other.data
        self._update_shape()
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self._check_other(other)
        out_data = self._data - other.data
        return self.__class__.from_array(self.gspc, out_data, copy_arr=False)

    def __isub__(self, other):
        self._check_other(other)
        self._data -= other.data
        self._update_shape()
        return self

    def __rsub__(self, other):
        self._check_other(other)
        out_data = other.data - self._data
        return self.__class__.from_array(self.gspc, out_data, copy_arr=False)

    def __mul__(self, other):
        self._check_other(other, check_subclass=True)
        out_data = self._data * other.data
        return self.__class__.from_array(self.gspc, out_data, copy_arr=False)

    def __rmul__(self, other):
        ndim = self._data.ndim - len(self.shape)
        axis = tuple(-(i + 1) for i in range(ndim))
        out_data = self.data * np.expand_dims(other, axis=axis)
        return self.__class__.from_array(self.gspc, out_data, copy_arr=False)

    def __imul__(self, other):
        if isinstance(other, Field):
            self._check_other(other, check_subclass=True)
            self._data *= other.data
        else:
            ndim = self._data.ndim - len(self.shape)
            axis = tuple(-(i + 1) for i in range(ndim))
            self._data *= np.expand_dims(other, axis=axis)
        self._update_shape()
        return self

    def __getitem__(self, item, copy=False):
        if isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise ValueError("too many indices: "
                                 f"field has {len(self.shape)} dimensions, "
                                 f"but {len(item)} were indexed.")
        return self.__class__.from_array(self.gspc, self._data[item],
                                         copy_arr=copy)


class GField(Field):

    def __init__(self, gspc: GSpace, shape: Union[int, tuple[int, ...]],
                 data: Optional[np.ndarray] = None):
        super().__init__(gspc, shape, data)

        data_shape = (*self.shape, self.gspc.numg)
        if data is None:
            data = np.empty(data_shape, dtype='c16', order='C')
        if data.shape != data_shape:
            raise ValueError(f"'data' must be a NumPy array of shape {data_shape}. "
                             f"got {data.shape}")
        self._data = data.astype(dtype='c16', order='C', copy=False)

    @property
    def g(self):
        return self._data

    @classmethod
    def from_array(cls, gspc: GSpace, arr: np.ndarray,
                   copy_arr: bool = True):
        if copy_arr:
            arr = np.copy(arr)
        shape = arr.shape[:-1]
        return cls(gspc, shape, arr)

    def _update_shape(self):
        self.shape = self._data.shape[:-1]

    def to_rfield(self):
        data_r = self.gspc.fft_mod.g2r(self._data)
        return RField.from_array(self.gspc, data_r, copy_arr=False)

    def symmetrize(self):
        self.gspc.symm_mod.symmetrize(self._data)


class RField(Field):

    def __init__(self, gspc: GSpace, shape: Union[int, tuple[int, ...]],
                 data: Optional[np.ndarray] = None):
        super().__init__(gspc, shape, data)

        data_shape = (*self.shape, *self.gspc.grid_shape)
        if data is None:
            data = np.empty(data_shape, dtype='c16', order='C')
        if data.shape != data_shape:
            raise ValueError(f"'data' must be a NumPy array of shape {data_shape}. "
                             f"got {data.shape}")
        self._data = data.astype(dtype='c16', order='C', copy=False)

    @property
    def r(self) -> np.ndarray:
        return self._data

    @classmethod
    def from_array(cls, gspc: GSpace, arr: np.ndarray,
                   copy_arr: bool = True):
        if copy_arr:
            arr = np.copy(arr)
        shape = arr.shape[:-3]
        return cls(gspc, shape, arr)

    def _update_shape(self):
        self.shape = self._data.shape[:-3]

    def to_gfield(self):
        data_g = self.gspc.fft_mod.r2g(self._data)
        return GField.from_array(self.gspc, data_g, copy_arr=False)

    def integrate(self, other=1, axis=None):
        if isinstance(other, Field):
            self._check_other(other)
            other = other.r
        out = np.sum(self._data * other, axis=(-1, -2, -3)) * self.gspc.reallat_dv
        if axis is None:
            return out
        else:
            return np.sum(out, axis=axis)
