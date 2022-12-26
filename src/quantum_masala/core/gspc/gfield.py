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
        self._r: np.ndarray = np.empty((*self.shape, *self.gspc.grid_shape), dtype='c16')

    def get_r(self, out=None):
        if out is None:
            out = np.empty((*self.shape, *self.gspc.grid_shape), dtype='c16')

        out[:] = self._r
        return out

    def set_r(self, arr):
        self._r[:] = arr

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, arr):
        self.set_r(arr)

    def get_g(self, out=None):
        if out is None:
            out = np.empty((*self.shape, self.gspc.numg), dtype='c16')

        self.gspc.fft_mod.r2g(self._r, out)
        return out

    def set_g(self, arr):
        self.gspc.fft_mod.g2r(arr, self._r)

    @property
    def g(self):
        return self.get_g()

    @g.setter
    def g(self, arr):
        self.set_g(arr)

    @classmethod
    def empty(cls, gspc: GSpace, shape: Union[int, tuple[int, ...]]) -> GField:
        return cls(gspc, shape)

    @classmethod
    def empty_like(cls, data: GField) -> GField:
        return cls(data.gspc, data.shape)

    @classmethod
    def zeros(cls, gspc: GSpace, shape: Union[int, tuple[int, ...]]) -> GField:
        out = cls(gspc, shape)
        out.r[:] = 0
        return out

    @classmethod
    def zeros_like(cls, data: GField) -> GField:
        return cls.zeros(data.gspc, data.shape)

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
            out.set_g(arr.reshape(*shape, gspc.numg))
        else:
            out.set_r(arr.reshape(*shape, *gspc.grid_shape))

        return out

    def copy(self):
        out = GField.empty_like(self)
        out.r = self._r
        return out

    def Bcast(self):
        self.pwcomm.world_comm.Bcast(self._r)

    def integrate_r(self, other=1):
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.r

        f_r = self._r * other
        return self.gspc.reallat_dv * np.sum(f_r, axis=(-1, -2, -3))

    def symmetrize(self):
        self.g = self.gspc.symm_mod.symmetrize(self.g)

    def __pos__(self):
        return self

    def __neg__(self):
        out = GField.empty_like(self)
        out.r = - self.r
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
            other = other.r
        data = self._r + other
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
            other = other.r

        self._r += other
        self.shape = self._r.shape[:-3]
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.r
        data = self._r - other
        return GField.from_array(self.gspc, data)

    def __isub__(self, other):
        if isinstance(other, GField):
            if other.gspc != self.gspc:
                raise ValueError(f"'gspc' do not match between the operands")
            other = other.r

        self._r -= other
        self.shape = self._r.shape[:-3]
        return self

    def __rmul__(self, other):
        """Multiplication defined for scalar and array-like objects

        Parameters
        ----------
        other

        Returns
        -------

        """
        return GField.from_array(self.gspc,
                                 np.expand_dims(other, axis=(-1, -2, -3)) * self._r)

    def __imul__(self, other):
        self._r *= np.expand_dims(other, axis=(-1, -2, -3))
        self.shape = self._r.shape[:-3]
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise ValueError("too many indices: "
                                 f"field has {len(self.shape)} dimensions, "
                                 f"but {len(item)} were indexed.")
        data = self._r[item]
        return GField.from_array(self.gspc, data)
