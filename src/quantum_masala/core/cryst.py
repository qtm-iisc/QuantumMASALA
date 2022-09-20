from dataclasses import dataclass, field
from typing import Type

import numpy as np
from numpy.typing import ArrayLike

from .ppdata import PseudoPotFile
from .constants import TPI

Vector3D = tuple[float, float, float]


class Lattice:
    """Class representing a lattice of translations

    Helper routines for coordinate transformations are implemented as methods

    Attributes
    ----------
        primvec : np.ndarray
            3X3 Array; Each **column** represents a primitive translation vector in atomic units.
        primvec_inv : np.ndarray
            Matrix Inverse of `primvec`.
        metric : np.ndarray
            Metric tensor of the Lattice Space.
        cellvol : float
            Volume of unit cell.
    """

    primvec: np.ndarray
    primvec_inv: np.ndarray
    metric: np.ndarray
    cellvol: float

    def __init__(self, primvec):
        """`Lattice` Constructor

        Parameters
        ----------
        primvec : array_like
            A 3X3 array whose columns represent the cartesian coords
            of lattice translation vectors in Hartree atomic units.
        """

        self.primvec = np.zeros((3, 3), dtype="f8", order="C")
        self.primvec[:] = primvec

        try:
            self.primvec_inv = np.linalg.inv(self.primvec)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "failed to invert 'primvec'. Axis Vectors are not linearly independent"
            )

        self.metric = self.primvec.T @ self.primvec
        self.cellvol = np.linalg.det(self.primvec)

    @property
    def axes_cart(self) -> list[np.ndarray]:
        """Returns list of vectors representing axes of lattice in cartesian coords"""
        return list(self.primvec.T)

    def cart2cryst(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector components in cartesian coords to crystal coords.

        Parameters
        ----------
        arr : array_like
            Input array of vector components in cartesian coords.
        axis : np.ndarray
            Axis indexing the 'i'th coordinate in `arr`.
            `arr.shape[axis]` must be equal to the dimension of the system, which is 3.

        Returns
        -------
            NumPy array with the same shape as `arr` containing the vectors in crystal coords.
        """
        vec_ = np.expand_dims(arr, axis)
        mat_ = self.primvec_inv.reshape((3, 3) + (1,) * (arr.ndim - axis - 1))
        return np.sum(mat_ * vec_, axis=axis + 1)

    def cryst2cart(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector components in crystal coords to crystal coords.

        Parameters
        ----------
        arr : array_like
            Input array of vector components in crystal coords.
        axis : np.ndarray
            Axis indexing the 'i'th coordinate in `arr`.
            `arr.shape[axis]` must be equal to the dimension of the system, which is 3.

        Returns
        -------
            NumPy Array with the same shape as `arr` containing the vectors in cartesian coords.
        """
        vec_ = np.expand_dims(arr, axis)
        mat_ = self.primvec.reshape((3, 3) + (1,) * (arr.ndim - axis - 1))
        return np.sum(mat_ * vec_, axis=axis + 1)

    def dot(self, l_vec1: np.ndarray, l_vec2: np.ndarray, coords: str = 'cryst'):
        """Computes the dot product between two sets of vectors given in either cartesian or crystal coords.

        Parameters
        ----------
        l_vec1, l_vec2 : np.ndarray
            Input Array of vector components. Their first axes must be of length 3.
        coords : {'cryst', 'cart'}; optional
            Coordinate type of input vector components.

        Returns
        -------
        Array containing dot product between vectors in `l_vec1` and `l_vec2`
        of shape `(l_vec1.shape[1:], l_vec2.shape[1:])`.
        """
        if l_vec1.shape[0] != 3 or l_vec2.shape[0] != 3:
            raise ValueError("Leading dimension of input arrays must be 3. "
                             f"Got {l_vec1.shape}, {l_vec2.shape}")

        shape_ = (*l_vec1.shape[1:], *l_vec2.shape[1:])
        l_vec1 = l_vec1.reshape(3, -1)
        l_vec2 = l_vec2.reshape(3, -1)
        if coords == 'cryst':
            return (l_vec1.T @ self.metric @ l_vec2).reshape(shape_)
        elif coords == 'cart':
            return (l_vec1.T @ l_vec2).reshape(shape_)
        else:
            raise ValueError(f"'coords' must be either 'cryst' or 'cart'. Got {coords}")

    def norm2(self, l_vec: np.ndarray, coords: str = 'cryst'):
        """Computes the norm squared of given vectors in either cartesian or crystal coords.

        Parameters
        ----------
        l_vec : np.ndarray
            Input Array of vector components. First axis must be of length 3.
        coords : {'cryst', 'cart'}; optional
            Coordinate type of vector components.

        Returns
        -------
        Array containing norm squared of vectors given in `l_vec`.
        """
        if l_vec.shape[0] != 3:
            raise ValueError("Leading dimension of input array must be 3. "
                             f"Got {l_vec.shape}")

        shape_ = l_vec.shape[1:]
        if coords == 'cryst':
            return np.sum(l_vec * (self.metric @ l_vec), axis=0).reshape(shape_)
        elif coords == 'cart':
            return np.sum(l_vec * l_vec, axis=0).reshape(shape_)
        else:
            raise ValueError(f"'coords' must be either 'cryst' or 'cart'. Got {coords}")

    def norm(self, l_vec: np.ndarray, coords: str = 'cryst'):
        """Computes the norm of given vectors in either cartesian or crystal coords.

        Parameters
        ----------
        l_vec : np.ndarray
            Input Array of vector components. First axis must be of length 3.
        coords : {'cryst', 'cart'}; optional
            Coordinate type of vector components.

        Returns
        -------
        Array containing norm squared of vectors given in `l_vec`.
        """
        return np.sqrt(self.norm2(l_vec, coords))


class RealLattice(Lattice):
    """Represents Real-Space Lattice of a Crystal.

    Extends `Lattice` with attribute aliases and methods for transformation to and from 'alat' coords.

    Attributes
    ----------
    alat : float
        Real-space Lattice Parameter 'a'
    latvec : np.ndarray
        Alias of `super().primvec`
    latvec_inv : np.ndarray
        Alias of `super().primvec_inv`
    adot : np.ndarray
        Alias of `super().metric`
    """

    alat: float
    latvec: np.ndarray
    latvec_inv: np.ndarray
    adot: np.ndarray

    def __init__(self, alat: float, latvec: np.ndarray):
        """Class constructor

        Parameters
        ----------
        alat : float
            Lattice parameter 'a'
        latvec : array_like
            3X3 Array representing the primitive translation vectors of the real-space lattice
        """
        self.alat: float = alat
        latvec = np.array(latvec)
        super().__init__(latvec)
        self.latvec = self.primvec
        self.latvec_inv = self.primvec_inv
        self.adot = self.metric

    @classmethod
    def from_axes_cart(cls, alat: float, a1: Vector3D, a2: Vector3D, a3: Vector3D):
        latvec = np.array([a1, a2, a3]).T
        return cls(alat, latvec)

    @classmethod
    def from_axes_alat(cls, alat: float, a1: Vector3D, a2: Vector3D, a3: Vector3D):
        latvec = alat * np.array([a1, a2, a3]).T
        return cls(alat, latvec)

    @property
    def axes_alat(self) -> list[np.ndarray]:
        """List of vectors representing axes of lattice in units of 'alat'"""
        return list(self.latvec.T / self.alat)

    def cart2alat(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector coords in atomic units to 'alat' units"""
        return np.array(arr) / self.alat

    def alat2cart(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector coords in 'alat' units to atomic units"""
        return np.array(arr) * self.alat

    def cryst2alat(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector components in crystal coords to cartesian coords ('alat' units)"""
        return self.cryst2cart(arr, axis) / self.alat

    def alat2cryst(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector components in cartesian coords ('alat' units) to crystal coords"""
        return self.cart2cryst(arr, axis) * self.alat

    def dot(self, l_vec1: np.ndarray, l_vec2: np.ndarray, coords: str = 'cryst'):
        """Same as `super().dot()` method, but exdending it for `alat`.

        Parameters
        ----------
        l_vec1, l_vec2 : np.ndarray
            Input Array of vector components. Their first axes must be of length 3.
        coords : {'cryst', 'cart', 'alat'}; optional
            Coordinate type of input vector components.

        Returns
        -------
        Array containing dot product between vectors in `l_vec1` and `l_vec2`
        of shape `(l_vec1.shape[1:], l_vec2.shape[1:])`.
        """
        if coords in ['cryst', 'cart']:
            return super().dot(l_vec1, l_vec2, coords)
        if coords == 'alat':
            return super().dot(l_vec1, l_vec2, 'cart') * self.alat**2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'alat'. Got {coords}")

    def norm2(self, l_vec: np.ndarray, coords: str = 'cryst'):
        """Same as `super().dot()` method, but exdending it for `alat`.

        Parameters
        ----------
        l_vec : np.ndarray
            Input Array of vector components. First axis must be of length 3.
        coords : {'cryst', 'cart', 'alat'}; optional
            Coordinate type of input vector components.

        Returns
        -------
        Array containing norm squared of vectors given in `l_vec`.
        """
        if coords in ['cryst', 'cart']:
            return super().norm2(l_vec, coords)
        if coords == 'alat':
            return super().norm2(l_vec, 'cart') * self.alat ** 2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'alat'. Got {coords}")


class ReciprocalLattice(Lattice):
    r"""Represents Reciprocal-Space Lattice of a Crystal.

   Extends `Lattice` with attribute aliases and methods for transformation to and from 'tpiba' coords.

    Attributes
    ----------
    tpiba: float
        Reciprocal-space Lattice Parameter :math:`2\pi / a`
    recvec: np.ndarray
        Alias of `super().primvec`
    recvec_inv: np.ndarray
        Alias of `super().primvec_inv`
    bdot : np.ndarray
        Alias of `super().metric`
    """

    tpiba: float
    recvec: np.ndarray
    recvec_inv: np.ndarray
    bdot: np.ndarray

    def __init__(self, tpiba: float, recvec: np.ndarray):
        r"""Class constructor.

        Parameters
        ----------
        tpiba : float
            Lattice parameter 'b' which equals :math:`2\pi / a`.
        recvec : array_like
            3X3 Array representing the primitive translation vectors of the reciprocal-space lattice.
        """

        self.tpiba = tpiba
        recvec = np.array(recvec)
        super().__init__(recvec)
        self.recvec = self.primvec
        self.recvec_inv = self.primvec_inv
        self.bdot = self.metric

    @classmethod
    def from_axes_cart(cls, tpiba: float, b1: Vector3D, b2: Vector3D, b3: Vector3D):
        recvec = np.array([b1, b2, b3]).T
        return cls(tpiba, recvec)

    @classmethod
    def from_axes_tpiba(cls, tpiba: float, b1: Vector3D, b2: Vector3D, b3: Vector3D):
        recvec = tpiba * np.array([b1, b2, b3]).T
        return cls(tpiba, recvec)

    @classmethod
    def from_reallat(cls, reallat: RealLattice):
        """Factory Method to construct from `RealLattice` instance.

        Parameters
        ----------
        reallat : RealLattice

        Returns
        -------
        `ReciprocalLattice` instance representing the reciprocal space of the crystal described by `reallat`
        """
        tpiba = TPI / reallat.alat
        recvec = TPI * np.transpose(reallat.latvec_inv)
        return cls(tpiba, recvec)

    @property
    def axes_tpiba(self) -> list[np.ndarray]:
        """List of vectors representing axes of lattice in units of 'tpiba'"""
        return list(self.recvec.T / self.tpiba)

    def cart2tpiba(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector coords in atomic units to 'tpiba' units"""
        return np.array(arr) / self.tpiba

    def tpiba2cart(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector coords in 'tpiba' units to atomic units"""
        return np.array(arr) * self.tpiba

    def cryst2tpiba(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector components in crystal coords to cartesian coords ('tpiba' units)"""
        return self.cryst2cart(arr, axis) / self.tpiba

    def tpiba2cryst(self, arr, axis: int = 0) -> np.ndarray:
        """Converts array of vector components in cartesian coords ('tpiba' units) to crystal coords"""
        return self.cart2cryst(arr, axis) * self.tpiba

    def dot(self, l_vec1: np.ndarray, l_vec2: np.ndarray, coords: str = 'cryst'):
        """Same as `super().dot()` method, but exdending it for `tpiba`.

        Parameters
        ----------
        l_vec1, l_vec2 : np.ndarray
            Input Array of vector components. Their first axes must be of length 3.
        coords : {'cryst', 'cart', 'tpiba'}; optional
            Coordinate type of input vector components.

        Returns
        -------
        Array containing dot product between vectors in `l_vec1` and `l_vec2`
        of shape `(l_vec1.shape[1:], l_vec2.shape[1:])`.
        """
        if coords in ['cryst', 'cart']:
            return super().dot(l_vec1, l_vec2, coords)
        if coords == 'alat':
            return super().dot(l_vec1, l_vec2, 'cart') * self.tpiba**2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'tpiba'. Got {coords}")

    def norm2(self, l_vec: np.ndarray, coords: str = 'cryst'):
        """Same as `super().dot()` method, but exdending it for `tpiba`.

        Parameters
        ----------
        l_vec : np.ndarray
            Input Array of vector components. First axis must be of length 3.
        coords : {'cryst', 'cart', 'tpiba'}; optional
            Coordinate type of input vector components.

        Returns
        -------
        Array containing norm squared of vectors given in `l_vec`.
        """
        if coords in ['cryst', 'cart']:
            return super().norm2(l_vec, coords)
        if coords == 'alat':
            return super().norm2(l_vec, 'cart') * self.tpiba ** 2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'tpiba'. Got {coords}")


@dataclass
class AtomBasis:
    """Class representing all the atoms of a given species in a unit cell

    Attributes
    ----------
    reallat: RealLattice
        Lattice of the Crystal
    ppdata: Type[PseudoPotFile]
        Pseudopotential Data of the species
    label: str
        Labe of the species given in `ppdata`
    mass: float
        Atomic mass of the species
    numatoms: int
        Number of atoms of given type per unit cell
    l_pos_cryst: np.ndarray
        Crystal Coords of atoms in the unit cell
    """

    reallat: RealLattice
    ppdata: Type[PseudoPotFile]
    mass: float

    cryst: ArrayLike

    label: str = field(init=False)
    numatoms: int = field(init=False)

    def __post_init__(self):
        self.label = self.ppdata.label

        if not isinstance(self.cryst, list):
            raise ValueError("Input argument 'cryst' must be a list of 3d vectors")
        self.cryst = np.array(self.cryst, dtype='f8', order='F').T

        if self.cryst.ndim != 2:
            raise ValueError(f"'cryst' must be a 2D Array. Got {self.cryst.shape}")
        if self.cryst.shape[0] != 3:
            raise ValueError(f"first axis of 'cryst' must be 3. Got {self.cryst.shape[0]}")
        self.numatoms = self.cryst.shape[1]

    @property
    def cart(self) -> np.ndarray:
        """Cartesian Coords of the atoms in atomic units"""
        return self.reallat.cryst2cart(self.cryst)

    @property
    def alat(self) -> np.ndarray:
        """Cartesian Coords of the atoms in 'alat' units"""
        return self.reallat.cryst2alat(self.cryst)


@dataclass
class Crystal:
    """Class representing the Crystal

    Attributes
    ----------
    reallat: RealLattice
        Real-Space Lattice of the crystal
    recilat: ReciprocalLattice
        Reciprocal-Space lattice of the crystal
    l_species: list[AtomBasis]
        List of atoms of different species representing the basis of the crystal.
    spglib_cell: tuple[np.ndarray, np.ndarray, tuple[int, ...]]
        A tuple containing crystal structure information supported by Spglib >= 1.9.1.

    Notes
    -----
    Please refer to `Spglib for Python` page for information regarding the `spglib_cell`:
    https://spglib.github.io/spglib/python-spglib.html#crystal-structure-cell
    """

    reallat: RealLattice
    l_species: list[AtomBasis]

    recilat: ReciprocalLattice = field(init=False)
    spglib_cell: tuple[np.ndarray, np.ndarray, tuple[int, ...]] = field(init=False)

    def __post_init__(self):
        self.recilat = ReciprocalLattice.from_reallat(self.reallat)

        lattice = self.reallat.latvec.T
        positions = [sp.cryst.T for sp in self.l_species]
        numbers = sum(((isp,) * len(pos) for isp, pos in enumerate(positions)), ())
        positions = np.concatenate(positions, axis=1)
        self.spglib_cell = (lattice, positions, numbers)
