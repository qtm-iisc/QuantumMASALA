from __future__ import annotations

__all__ = ["GField"]

from typing import Union, Optional
import numpy as np

from quantum_masala import config

from .gspc import GSpace


class GField:

    def __init__(self, gspc: GSpace, shape: Union[int, tuple[int, ...]]):
        self.pwcomm = config.pwcomm
        self.gspc: GSpace = gspc

        if isinstance(shape, int):
            shape = (shape, )
        self.shape: tuple[int, ...] = shape

        self._g: np.ndarray = np.empty((*self.shape, self.gspc.numg), dtype='c16')

    @classmethod
    def empty(cls, gspc: GSpace, shape: Union[int, tuple[int, ...]]) -> GField:
        return cls(gspc, shape)

    @classmethod
    def empty_like(cls, data: GField) -> GField:
        return cls(data.gspc, data.shape)

    @classmethod
    def zeros(cls, gspc: GSpace, shape: Union[int, tuple[int, ...]]) -> GField:
        out = cls(gspc, shape)
        out.g[:] = 0
        return out

    @classmethod
    def zeros_like(cls, data: GField) -> GField:
        out = cls(data.gspc, data.shape)
        out.g[:] = 0
        return out

    @classmethod
    def from_array(cls, gspc: GSpace, arr: np.ndarray):
        if arr.shape[-1] == gspc.numg:
            spc = 'g'
            shape = arr.shape[:-1]
        elif arr.shape[-3:] == gspc.grid_shape:
            spc = 'r'
            shape = arr.shape[:-3]
        else:
            raise ValueError(f"shape of 'arr' not compatible with GSpace 'gspc': "
                             f"Expected {(..., gspc.numg)} or {(..., *gspc.grid_shape)}, "
                             f"Got {arr.shape}")
        if shape == ():
            shape = (1, )

        out = cls(gspc, shape)
        if spc == 'g':
            out.set_g(arr)
        else:
            out.set_r(arr)

        return out

    def get_g(self, out=None):
        if out is None:
            out = np.empty((*self.shape, self.gspc.numg), dtype='c16')

        out[:] = self._g
        return out

    def set_g(self, arr):
        self._g[:] = arr

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, arr):
        self.set_g(arr)

    def get_r(self, out=None):
        if out is None:
            out = np.empty((*self.shape, *self.gspc.grid_shape), dtype='c16')

        self.gspc.fft_mod.g2r(self._g, out)
        return out

    def set_r(self, arr):
        self.gspc.fft_mod.r2g(np.array(arr, dtype='c16'), self._g)

    @property
    def r(self):
        return self.get_r()

    @r.setter
    def r(self, arr):
        self.set_r(arr)

    def copy(self):
        out = GField.empty_like(self)
        out.g[:] = self._g
        return out

    def Bcast(self):
        self.pwcomm.world_comm.Bcast(self._g)

    def integrate_r(self, other=1):
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.r

        f_r = self.r * other
        return self.gspc.reallat_dv * np.sum(f_r, axis=(-1, -2, -3))

    def symmetrize(self):
        self.gspc.symm_mod.symmetrize(self._g)

    def __pos__(self):
        return self

    def __neg__(self):
        out = GField.empty_like(self)
        out.g[:] = - self._g
        return out

    def __add__(self, other):
        """Addition defined for other ``GField`` instances, scalar and
        array-like objects

        Parameters
        ----------
        other
            Operand to add

        Returns
        -------
        ``GField`` instance representing the sum of the operands
        """
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.g
        data = self._g + other
        return GField.from_array(self.gspc, data)

    def __iadd__(self, other):
        """In-place addition. Can also update shape based on NumPy broadcasting
        rules

        Parameters
        ----------
        other
            Operand to add

        Returns
        -------

        """
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.g

        self._g += other
        self.shape = self._g.shape[:-1]
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.g
        data = self._g - other
        return GField.from_array(self.gspc, data)

    def __isub__(self, other):
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.g

        self._g -= other
        self.shape = self._g.shape[:-1]
        return self

    def __rmul__(self, other):
        """Multiplication defined for scalar and array-like objects

        Parameters
        ----------
        other

        Returns
        -------

        """
        return GField.from_array(self.gspc, np.expand_dims(other, axis=-1) * self._g)

    def __imul__(self, other):
        self._g *= np.expand_dims(other, axis=-1)
        self.shape = self._g.shape[:-1]
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise ValueError("too many indices: "
                                 f"field has {len(self.shape)} dimensions, "
                                 f"but {len(item)} were indexed.")
        data = self._g[item]
        return GField.from_array(self.gspc, data)
