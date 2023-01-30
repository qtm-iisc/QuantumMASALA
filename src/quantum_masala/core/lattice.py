"""Module for describing crystal lattices in QuantumMASALA

This module contains definitions for ``Lattice`` class and its derivatives that
is used across QuantumMASALA to represent crystal lattices. The classes
contain many methods like transforming coordinates between cartesian and
crystal, computing dot products between lists of vectors, etc.

TODO : Complete docs for ``ReciprocalLattice``
"""

__all__ = ["Lattice", "RealLattice", "ReciprocalLattice"]

import numpy as np
from spglib import get_symmetry

from quantum_masala.constants import TPI, ANGSTROM_BOHR


class Lattice:
    """Represents a lattice of translations

    Describes a lattice via its primitive translation vectors and provides
    simple methods for transformation of vector components.
    Note that this class is not used as-is and is inherited by
    ``RealLattice`` and ``ReciprocalLattice``.

    Parameters
    ----------
    primvec : numpy.ndarray
        (``(3, 3)``) 2D array where each **column** represents a primitive
        translation vector.

    Raises
    ------
    ValueError
        Raised if ``primvec`` is singular, implying that the vectors are
        not linearly independent.
    """

    def __init__(self, primvec: np.ndarray):
        self.primvec: np.ndarray = np.array(primvec, dtype='f8', order='C')
        """(``(3, 3)``, ``'f8'``, ``'C'``) Matrix representing
        lattice translation vectors.
        """

        try:
            primvec_inv = np.linalg.inv(self.primvec)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "failed to invert 'primvec'. Axis Vectors are not linearly independent"
            )
        self.primvec_inv: np.ndarray = primvec_inv
        """(``(3, 3)``) Matrix inverse of ``primvec``.
        """

        self.metric: np.ndarray = self.primvec.T @ self.primvec
        """(``(3, 3)``) Metric Tensor.
        """
        self.cellvol: float = np.linalg.det(self.primvec)
        """Volume of unit cell.
        """

    @property
    def axes_cart(self) -> list[np.ndarray]:
        """List of the three primitive vectors inatomic units
        """
        return list(self.primvec.T)

    def cart2cryst(self, arr: np.ndarray, axis: int = 0):
        """Transforms array of vector components in cartesian coords
        to crystal coords.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array of vector components in cartesian coords.
        axis : int, optional
            Axis indexing the ``i``th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        arr_cryst : numpy.ndarray
            Array with the same shape as ``arr`` containing
            the vectors in crystal coords.
        """
        arr = np.asarray(arr)
        vec_ = np.expand_dims(arr, axis)
        mat_ = self.primvec_inv.reshape((3, 3) + (1,) * (arr.ndim - axis - 1))
        return np.sum(mat_ * vec_, axis=axis + 1)

    def cryst2cart(self, arr: np.ndarray, axis: int = 0):
        """Transforms array of vector components in crystal coords
        to cartesian coords.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array of vector components in crystal coords.
        axis : int, optional
            Axis indexing the ``i``th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        arr_cart : numpy.ndarray
            Array with the same shape as ``arr`` containing
            the vectors in cartesian coords.
        """
        arr = np.asarray(arr)
        vec_ = np.expand_dims(arr, axis)
        mat_ = self.primvec.reshape((3, 3) + (1,) * (arr.ndim - axis - 1))
        return np.sum(mat_ * vec_, axis=axis + 1)

    def dot(self, l_vec1: np.ndarray, l_vec2: np.ndarray, coords: str = 'cryst'):
        """Computes the dot product between two sets of vectors.

        Parameters
        ----------
        l_vec1, l_vec2 : numpy.ndarray
            Input Array of vector components.
            Their first axes must have length 3.
        coords : {'cryst', 'cart'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        arr_dot : numpy.ndarray
            Array of dot products between vectors in ``l_vec1`` and
            ``l_vec2`` of shape (``l_vec1.shape[1:]``, ``l_vec2.shape[1:]``).

        Raises
        ------
        ValueError
            Raised if arrays are of improper shape.
        ValueError
            Raised if value of ``coords`` is invalid.
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
        """Computes the norm squared of given vectors.

        Parameters
        ----------
        l_vec : numpy.ndarray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        arr_norm2 : numpy.ndarray
            Array containing norm squared of vectors given in ``l_vec``.

        Raises
        ------
        ValueError
            Raised if ``l_vec`` is of improper shape.
        ValueError
            Raised if value of ``coords`` is invalid.
        """
        if l_vec.shape[0] != 3:
            raise ValueError("Leading dimension of input array must be 3. "
                             f"Got {l_vec.shape}")
        l_vec = l_vec.reshape(3, -1)
        shape_ = l_vec.shape[1:]
        if coords == 'cryst':
            return np.sum(l_vec * (self.metric @ l_vec), axis=0).reshape(shape_)
        elif coords == 'cart':
            return np.sum(l_vec * l_vec, axis=0).reshape(shape_)
        else:
            raise ValueError(f"'coords' must be either 'cryst' or 'cart'. Got {coords}")

    def norm(self, l_vec: np.ndarray, coords: str = 'cryst'):
        """Computes the norm of given vectors.

        Parameters
        ----------
        l_vec : numpy.ndarray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        arr_norm : numpy.ndarray
            Array containing norm of vectors given in ``l_vec``.

        Raises
        ------
        ValueError
            Raised if ``l_vec`` is of improper shape.
        ValueError
            Raised if value of ``coords`` is invalid.
        """
        return np.sqrt(self.norm2(l_vec, coords))

    def get_symmetry(self):
        lattice = np.transpose(self.primvec)
        positions = np.zeros((1, 3))
        numbers = np.zeros(1, dtype='i8')

        symm = get_symmetry((lattice, positions, numbers))
        del symm['equivalent_atoms']
        return symm


class RealLattice(Lattice):
    """Represents Real-Space Lattice of a Crystal.

    Extends ``Lattice`` with attribute aliases and methods for
    transformation to and from 'alat' coords.

    Parameters
    ----------
    alat : float
        Lattice parameter 'a'
    latvec : numpy.ndarray
        (``(3, 3)``) 2D array where each **column** represents a primitive
        translation vector.
    """

    def __init__(self, alat: float, latvec: np.ndarray):
        self.alat: float = alat
        """Lattice parameter 'a'
        """
        latvec = np.array(latvec)
        super().__init__(latvec)
        self.latvec: np.ndarray = self.primvec
        """(``(3, 3)``, ``'f8'``) Alias of ``self.primvec``
        """
        self.latvec_inv: np.ndarray = self.primvec_inv
        """(``(3, 3)``, ```'f8'``) Alias of ``self.primvec_inv``
        """
        self.adot: np.ndarray = self.metric
        """(``(3, 3)``, ```'f8'``) Alias of ``self.metric``
        """

    @classmethod
    def from_bohr(cls, alat: float, a1, a2, a3):
        """Generates ``RealLattice`` instance from a list of
        primitive translation vectors in atomic units

        Parameters
        ----------
        alat : float
            Bravais Lattice parameter 'a' in Bohr
        a1, a2, a3 : array-like
            (``(3, )``) Cartesian components of translation vectors in Bohr

        Returns
        ------
        reallat : RealLattice
            Represents a real lattice whose primitive translation axes are
            given by ``a1``, ``a2`` and ``a3``
        """
        latvec = np.array([a1, a2, a3]).T
        return cls(alat, latvec)

    @classmethod
    def from_angstrom(cls, alat: float, a1, a2, a3):
        """Generates ``RealLattice`` instance from a list of
        primitive translation vectors in angstrom

        Parameters
        ----------
        alat : float
            Bravais Lattice parameter 'a' in angstrom
        a1, a2, a3 : array-like
            (``(3, )``) Cartesian components of translation vectors in angstrom

        Returns
        ------
        reallat : `RealLattice`
            Represents a real lattice whose primitive translation axes are
            given by ``a1``, ``a2`` and ``a3``
        """
        alat = alat * ANGSTROM_BOHR
        latvec = np.array([a1, a2, a3]).T * ANGSTROM_BOHR
        return cls(alat, latvec)

    @classmethod
    def from_alat(cls, alat: float, a1, a2, a3):
        """Generates ``RealLattice`` instance from a list of
        primitive translation vectors in 'alat' units

        Parameters
        ----------
        alat : float
            Bravais Lattice parameter 'a' in Bohr
        a1, a2, a3 : array-like
            (``(3, )``) Cartesian components of translation vectors in 'alat'

        Returns
        ------
        reallat : RealLattice
            Represents a real lattice whose primitive translation axes are
            given by ``a1``, ``a2`` and ``a3``
        """
        latvec = alat * np.array([a1, a2, a3]).T
        return cls(alat, latvec)

    @classmethod
    def from_recilat(cls, recilat: 'ReciprocalLattice'):
        """Constructs the real lattice dual to the reciprocal lattice

        Parameters
        ----------
        recilat : ReciprocalLattice
            Reciprocal Lattice

        Returns
        -------
        reallat : RealLattice
            Real Lattice that is the dual of Reciprocal Lattice ``recilat``
        """
        alat = TPI / recilat.tpiba
        latvec = np.transpose(recilat.recvec_inv) * TPI
        return cls(alat, latvec)

    @property
    def axes_alat(self) -> list[np.ndarray]:
        """List of the three primitive vectors in 'alat' units
        """
        return list(self.latvec.T / self.alat)

    def cart2alat(self, arr: np.ndarray, axis: int = 0):
        """Transforms array of vector components in atomic units
        to 'alat' units.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array of vector components in atomic units.
        axis : int, optional
            Axis indexing the `i`th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        arr_alat : numpy.ndarray
            Array with the same shape as ``arr`` containing
            the vectors in 'alat' units.
        """
        return np.array(arr) / self.alat

    def alat2cart(self, arr: np.ndarray, axis: int = 0):
        """Transforms array of vector components in 'alat' units
        to atomic units.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array of vector components in 'alat' units
        axis : int, optional
            Axis indexing the `i`th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        arr_cart : numpy.ndarray
            Array with the same shape as ``arr`` containing
            the vectors in atomic units.
        """
        return np.array(arr) * self.alat

    def cryst2alat(self, arr: np.ndarray, axis: int = 0):
        """Transforms array of vector components in crystal coords
        to cartesian coords with 'alat' units.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array of vector components in crystal coords.
        axis : int, optional
            Axis indexing the ``i``th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        arr_alat : numpy.ndarray
            Array with the same shape as ``arr`` containing
            the vectors in cartesian coords with 'alat' units.
        """
        return self.cryst2cart(arr, axis) / self.alat

    def alat2cryst(self, arr: np.ndarray, axis: int = 0):
        """Transforms array of vector components in cartesian coords
        with 'alat' units to crystal coords.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array of vector components in 'alat' units.
        axis : int, optional
            Axis indexing the ``i``th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        arr_cryst : numpy.ndarray
            Array with the same shape as ``arr`` containing
            the vectors in crystal coords.
        """
        return self.cart2cryst(arr, axis) * self.alat

    def dot(self, l_vec1: np.ndarray, l_vec2: np.ndarray, coords: str = 'cryst'):
        """Same as ``Lattice.dot()`` method, but exdending it for
        ``coords='alat'``.

        Parameters
        ----------
        l_vec1, l_vec2 : numpy.ndarray
            Input Array of vector components.
            Their first axes must have length 3.
        coords : {'cryst', 'cart', 'alat'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        arr_dot : numpy.ndarray
            Array of dot products between vectors in ``l_vec1`` and
            ``l_vec2`` of shape (``l_vec1.shape[1:]``, ``l_vec2.shape[1:]``).

        Raises
        ------
        ValueError
            Raised if arrays are of improper shape.
        ValueError
            Raised if value of ``coords`` is invalid.
        """
        if coords in ['cryst', 'cart']:
            return super().dot(l_vec1, l_vec2, coords)
        if coords == 'alat':
            return super().dot(l_vec1, l_vec2, 'cart') * self.alat**2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'alat'. Got {coords}")

    def norm2(self, l_vec: np.ndarray, coords: str = 'cryst'):
        """Same as ``Lattice.norm2()`` method, but exdending it for
        ``coords='alat'``.

        Parameters
        ----------
        l_vec : numpy.ndarray
            (``(3, ...)``) Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart', 'alat'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        arr_norm2 : numpy.ndarray
            (``(l_vec.shape[1:]``) Array containing norm squared of vectors
            given in `l_vec`.

        Raises
        ------
        ValueError
            Raised if ``l_vec`` is of improper shape.
        ValueError
            Raised if value of ``coords`` is invalid.
        """
        if coords in ['cryst', 'cart']:
            return super().norm2(l_vec, coords)
        if coords == 'alat':
            return super().norm2(l_vec, 'cart') * self.alat ** 2
        else:
            raise ValueError("'coords' must be one of 'cryst', 'cart' or 'alat'. "
                             f"Got {coords}")

    def norm(self, l_vec: np.ndarray, coords: str = 'cryst'):
        """Computes the norm of given vectors.

        Parameters
        ----------
        l_vec : numpy.ndarray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart', 'alat'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        arr_norm : numpy.ndarray
            Array containing norm of vectors given in `l_vec`.

        Raises
        ------
        ValueError
            Raised if ``l_vec`` is of improper shape.
        ValueError
            Raised if value of ``coords`` is invalid.
        """
        if coords in ['cryst', 'cart']:
            return super().norm(l_vec, coords)
        if coords == 'alat':
            return super().norm(l_vec, 'cart') * self.alat
        else:
            raise ValueError("'coords' must be one of 'cryst', 'cart' or 'alat'. "
                             f"Got {coords}")

    def get_mesh_coords(self, n1: int = 1, n2: int = 1, n3: int = 1,
                        coords: str = 'cryst'):
        xi = []
        for i, n in enumerate([n1, n2, n3]):
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"'n{i+1}' must be a positive integer. "
                                 f"got {n} (type {type(n)})")
            xi.append(np.fft.fftfreq(n))
        r_cryst = np.array(np.meshgrid(*xi, indexing='ij'))
        if coords == 'cryst':
            return r_cryst
        elif coords == 'cart':
            return self.cryst2cart(r_cryst)
        elif coords == 'alat':
            return self.cryst2alat(r_cryst)


class ReciprocalLattice(Lattice):
    """Represents Reciprocal-Space Lattice of a Crystal.

    Extends ``Lattice`` with attribute aliases and methods for
    transformation to and from 'tpiba' coords.

    Parameters
    ----------
    tpiba : float
        Lattice parameter 'tpiba'
    recvec: numpy.ndarray
        (``(3, 3)``) 2D array where each **column** represents a primitive
        translation vector.
    """

    def __init__(self, tpiba: float, recvec: np.ndarray):
        self.tpiba: float = tpiba
        """Lattice parameter 'tpiba' (`float`)
        """
        recvec = np.array(recvec)
        super().__init__(recvec)
        self.recvec: np.ndarray = self.primvec
        """(``(3, 3)``, ``'f8'``) Alias of ``self.primvec``
        """
        self.recvec_inv: np.ndarray = self.primvec_inv
        """(``(3, 3)``, ``'f8'``) Alias of ``self.primvec_inv``
        """
        self.bdot: np.ndarray = self.metric
        """(``(3, 3``, ``'f8'``) Alias of ``self.metric``
        """

    @classmethod
    def from_cart(cls, tpiba: float, b1, b2, b3):
        """Generates ``ReciprocalLattice`` instance from a list of
        primitive translation vectors in atomic units

        Parameters
        ----------
        tpiba : float
            Bravais Lattice parameter 'tpiba' in Bohr^-1
        b1, b2, b3 : array-like
            Cartesian components of translation vectors in Bohr^-1

        Returns
        ------
        recilat : ReciprocalLattice
            Represents a reciprocal lattice whose primitive translation axes
            are given by ``b1``, ``b2`` and ``b3``
        """
        recvec = np.array([b1, b2, b3]).T
        return cls(tpiba, recvec)

    @classmethod
    def from_tpiba(cls, tpiba: float, b1, b2, b3):
        """Generates ``ReciprocalLattice`` instance from a list of
        primitive translation vectors in 'tpiba' units

        Parameters
        ----------
        tpiba : float
            Lattice parameter 'tpiba' in Bohr^-1
        b1, b2, b3 : array-like
            Cartesian components of translation vectors in 'tpiba'

        Returns
        ------
        recilat : ReciprocalLattice
            Represents a reciprocal lattice whose primitive translation axes
            are given by ``b1``, ``b2`` and ``b3``
        """
        recvec = tpiba * np.array([b1, b2, b3]).T
        return cls(tpiba, recvec)

    @classmethod
    def from_reallat(cls, reallat: RealLattice):
        """Constructs the reciprocal lattice dual to the real lattice

        Parameters
        ----------
        reallat : RealLattice
            Real Lattice

        Returns
        -------
        recilat : ReciprocalLattice
            Reciprocal Lattice that is the dual of Real Lattice ``reallat``
        """
        tpiba = TPI / reallat.alat
        recvec = TPI * np.transpose(reallat.latvec_inv)
        return cls(tpiba, recvec)

    @property
    def axes_tpiba(self) -> list[np.ndarray]:
        """List of vectors representing axes of lattice
        in units of 'tpiba'
        """
        return list(self.recvec.T / self.tpiba)

    def cart2tpiba(self, arr: np.ndarray, axis: int = 0):
        """Converts array of vector coords in atomic units to 'tpiba' units"""
        return np.array(arr) / self.tpiba

    def tpiba2cart(self, arr: np.ndarray, axis: int = 0):
        """Converts array of vector coords in 'tpiba' units to atomic units"""
        return np.array(arr) * self.tpiba

    def cryst2tpiba(self, arr: np.ndarray, axis: int = 0):
        """Converts array of vector components in crystal coords to cartesian coords ('tpiba' units)"""
        return self.cryst2cart(arr, axis) / self.tpiba

    def tpiba2cryst(self, arr: np.ndarray, axis: int = 0):
        """Converts array of vector components in cartesian coords ('tpiba' units) to crystal coords"""
        return self.cart2cryst(arr, axis) * self.tpiba

    def dot(self, l_vec1: np.ndarray, l_vec2: np.ndarray, coords: str = 'cryst'):
        """Same as `super().dot()` method, but exdending it for `tpiba`.

        Parameters
        ----------
        l_vec1, l_vec2 : numpy.ndarray
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
        if coords == 'tpiba':
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
        if coords == 'tpiba':
            return super().norm2(l_vec, 'cart') * self.tpiba ** 2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'tpiba'. Got {coords}")
