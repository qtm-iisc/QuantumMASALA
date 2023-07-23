# from __future__ import annotations
from typing import Self
from qtm.config import NDArray
__all__ = ['Lattice', 'RealLattice', 'ReciLattice', 'NDArray']

import numpy as np

from qtm.constants import TPI, ANGSTROM


class Lattice:
    """Represents the lattice of translations

    Describes a lattice by its primitive translation vectors and provides
    methods for coordinate transformations.
    Base class of `RealLattice` and `ReciprocalLatvec`.

    Parameters
    ----------
    primvec : NDArray
        (``(3, 3)``) 2D array where each **column** represents a primitive
        translation vector.

    Raises
    ------
    ValueError
        Raised if `primvec` is singular, i.e the basis vectors are
        not linearly independent.
    """
    def __init__(self, primvec: NDArray):
        if primvec.shape != (3, 3) or primvec.dtype != 'f8':
            raise ValueError(
                "'primvec' must be a 2D array with shape (3, 3) and dtype 'f8'. got: "
                f"shape={primvec.shape if hasattr(primvec, 'shape') else 'NA'}, "
                f"dtype={primvec.dtype if hasattr(primvec, 'dtype') else 'NA'}"
            )

        self.primvec: NDArray = primvec.copy('C')
        r"""(``(3, 3)``, ``'f8'``, ``'C'``) Matrix containing
        lattice translation vectors. 
        :math:`\vec{a}_i` is given by `primvec[:, i]`.
        """

        try:
            primvec_inv = np.linalg.inv(self.primvec)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "failed to invert 'primvec'; basis vectors are not linearly independent."
            ) from e
        self.primvec_inv: NDArray = primvec_inv
        """(``(3, 3)``) Matrix inverse of `primvec`.
        """

        self.metric: NDArray = self.primvec.T @ self.primvec
        """(``(3, 3)``) Metric Tensor.
        """
        self.cellvol: float = float(np.linalg.det(self.primvec))
        """Volume of unit cell.
        """

    @property
    def axes_cart(self) -> tuple[list[float], ...]:
        """tuple of the three primitive vectors in atomic units
        """
        return tuple(vec.tolist() for vec in self.primvec.T)

    def cart2cryst(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in cartesian coords
        to crystal coords.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in cartesian coords.
        axis : int, optional
            Axis indexing the vector coordinates/components.
            `arr.shape[axis]` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as `arr` containing
            the vectors in crystal coords.
        """
        vec_ = np.expand_dims(arr, axis)
        mat_ = self.primvec_inv.reshape((3, 3) + (1,) * (vec_.ndim - axis - 2))
        return np.sum(mat_ * vec_, axis=axis + 1)

    def cryst2cart(self, arr: NDArray, axis: int = 0):
        """Transforms array of vector components in crystal coords
        to cartesian coords.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in crystal coords.
        axis : int, optional
            Axis indexing the vector coordinates/components.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as ``arr`` containing
            the vectors in cartesian coords.
        """
        vec_ = np.expand_dims(arr, axis)
        mat_ = self.primvec.reshape((3, 3) + (1,) * (vec_.ndim - axis - 2))
        return np.sum(mat_ * vec_, axis=axis + 1)

    def dot(self, l_vec1: NDArray, l_vec2: NDArray, coords: str = 'cryst') -> NDArray:
        """Computes the dot product between two sets of vectors

        Parameters
        ----------
        l_vec1, l_vec2 : NDArray
            Input Array of vector components.
            Their first axes must have length 3.
        coords : {'cryst', 'cart'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
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

    def norm2(self, l_vec: NDArray, coords: str = 'cryst') -> NDArray:
        """Computes the norm squared of input vectors.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
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

    def norm(self, l_vec: NDArray, coords: str = 'cryst') -> NDArray:
        """Computes the norm of given vectors.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
            Array containing norm of vectors given in ``l_vec``.

        Raises
        ------
        ValueError
            Raised if ``l_vec`` is of improper shape.
        ValueError
            Raised if value of ``coords`` is invalid.
        """
        return np.sqrt(self.norm2(l_vec, coords))


class RealLattice(Lattice):
    """Represents Real-Space Lattice of a Crystal.

    Extends ``Lattice`` with attribute aliases and methods for
    transformation to and from 'alat' coords.

    Parameters
    ----------
    alat : float
        Lattice parameter 'a'
    latvec : NDArray
        (``(3, 3)``) 2D array where each **column** represents a primitive
        translation vector.
    """

    def __init__(self, alat: float, latvec: NDArray):
        self.alat: float = alat
        """Lattice parameter 'a'
        """
        super().__init__(latvec)
        self.latvec: NDArray = self.primvec
        """(``(3, 3)``, ``'f8'``) Alias of ``primvec``
        """
        self.latvec_inv: NDArray = self.primvec_inv
        """(``(3, 3)``, ```'f8'``) Alias of ``primvec_inv``
        """
        self.adot: NDArray = self.metric
        """(``(3, 3)``, ```'f8'``) Alias of ``metric``
        """

    @classmethod
    def from_bohr(cls, alat: float, a1, a2, a3) -> Self:
        """Generates ``RealLatvec`` instance from a list of
        primitive translation vectors in atomic units

        Parameters
        ----------
        alat : float
            Bravais Lattice parameter 'a' in Bohr
        a1, a2, a3 : array-like
            (``(3, )``) Cartesian components of translation vectors in Bohr

        Returns
        ------
        RealLattice
            Represents a real lattice whose primitive translation axes are
            given by ``a1``, ``a2`` and ``a3``
        """
        latvec = np.stack((a1, a2, a3), axis=1)
        return cls(alat, latvec)

    @classmethod
    def from_angstrom(cls, alat: float, a1, a2, a3) -> Self:
        """Generates ``RealLatvec`` instance from a list of
        primitive translation vectors in angstrom

        Parameters
        ----------
        alat : float
            Bravais Lattice parameter 'a' in angstrom
        a1, a2, a3 : array-like
            (``(3, )``) Cartesian components of translation vectors in angstrom

        Returns
        ------
        RealLattice
            Represents a real lattice whose primitive translation axes are
            given by ``a1``, ``a2`` and ``a3``
        """
        alat = alat * ANGSTROM
        latvec = np.stack((a1, a2, a3), axis=1) * ANGSTROM
        return cls(alat, latvec)

    @classmethod
    def from_alat(cls, alat: float, a1, a2, a3) -> Self:
        """Generates ``RealLatvec`` instance from a list of
        primitive translation vectors in 'alat' units

        Parameters
        ----------
        alat : float
            Bravais Lattice parameter 'a' in Bohr
        a1, a2, a3 : array-like
            (``(3, )``) Cartesian components of translation vectors in 'alat'

        Returns
        ------
        RealLattice
            Represents a real lattice whose primitive translation axes are
            given by ``a1``, ``a2`` and ``a3``
        """
        latvec = alat * np.stack((a1, a2, a3), axis=1)
        return cls(alat, latvec)

    @classmethod
    def from_recilat(cls, recilat: 'ReciLattice') -> Self:
        """Constructs the real lattice dual to the reciprocal lattice

        Parameters
        ----------
        recilat : ReciLattice
            Reciprocal Lattice

        Returns
        -------
        RealLattice
            Real Lattice that is the dual of Reciprocal Lattice ``recilat``
        """
        alat = TPI / recilat.tpiba
        latvec = recilat.recvec_inv.T * TPI
        return cls(alat, latvec)

    @property
    def axes_alat(self) -> tuple[list[float], ...]:
        """List of the three primitive vectors in 'alat' units
        """
        return tuple(vec.tolist() for vec in self.latvec.T / self.alat)

    def cart2alat(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in atomic units
        to 'alat' units.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in atomic units.
        axis : int, optional
            Axis indexing the `i`th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as ``arr`` containing
            the vectors in 'alat' units.
        """
        return arr / self.alat

    def alat2cart(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in 'alat' units
        to atomic units.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in 'alat' units
        axis : int, optional
            Axis indexing the `i`th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as ``arr`` containing
            the vectors in atomic units.
        """
        return arr * self.alat

    def cryst2alat(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in crystal coords
        to cartesian coords with 'alat' units.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in crystal coords.
        axis : int, optional
            Axis indexing the ``i``th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as ``arr`` containing
            the vectors in cartesian coords with 'alat' units.
        """
        return self.cryst2cart(arr, axis) / self.alat

    def alat2cryst(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in cartesian coords
        with 'alat' units to crystal coords.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in 'alat' units.
        axis : int, optional
            Axis indexing the ``i``th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as ``arr`` containing
            the vectors in crystal coords.
        """
        return self.cart2cryst(arr, axis) * self.alat

    def dot(self, l_vec1: NDArray, l_vec2: NDArray, coords: str = 'cryst') -> NDArray:
        """Same as ``Lattice.dot()`` method, but exdending it for
        ``coords='alat'``.

        Parameters
        ----------
        l_vec1, l_vec2 : NDArray
            Input Array of vector components.
            Their first axes must have length 3.
        coords : {'cryst', 'cart', 'alat'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
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

    def norm2(self, l_vec: NDArray, coords: str = 'cryst') -> NDArray:
        """Same as ``Lattice.norm2()`` method, but exdending it for
        ``coords='alat'``.

        Parameters
        ----------
        l_vec : NDArray
            (``(3, ...)``) Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart', 'alat'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
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

    def norm(self, l_vec: NDArray, coords: str = 'cryst') -> NDArray:
        """Computes the norm of given vectors.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart', 'alat'}, optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
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


class ReciLattice(Lattice):
    """Represents Reciprocal-Space Lattice of a Crystal.

    Extends ``Lattice`` with attribute aliases and methods for
    transformation to and from 'tpiba' coords.

    Parameters
    ----------
    tpiba : float
        Lattice parameter 'tpiba'
    recvec: NDArray
        (``(3, 3)``) 2D array where each **column** represents a primitive
        translation vector.
    """

    def __init__(self, tpiba: float, recvec: NDArray):
        self.tpiba: float = tpiba
        r"""Lattice parameter :math:`\frac{2\pi}{a}`
        """
        super().__init__(recvec)
        self.recvec: NDArray = self.primvec
        """(``(3, 3)``, ``'f8'``) Alias of ``primvec``
        """
        self.recvec_inv: NDArray = self.primvec_inv
        """(``(3, 3)``, ``'f8'``) Alias of ``primvec_inv``
        """
        self.bdot: NDArray = self.metric
        """(``(3, 3)``, ``'f8'``) Alias of ``metric``
        """

    @classmethod
    def from_cart(cls, tpiba: float, b1, b2, b3) -> Self:
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
        ReciprocalLatvec
            Represents a reciprocal lattice whose primitive translation axes
            are given by ``b1``, ``b2`` and ``b3``
        """
        recvec = np.array([b1, b2, b3]).T
        return cls(tpiba, recvec)

    @classmethod
    def from_tpiba(cls, tpiba: float, b1, b2, b3) -> Self:
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
        ReciprocalLatvec
            Represents a reciprocal lattice whose primitive translation axes
            are given by ``b1``, ``b2`` and ``b3``
        """
        recvec = tpiba * np.array([b1, b2, b3]).T
        return cls(tpiba, recvec)

    @classmethod
    def from_reallat(cls, reallat: RealLattice) -> Self:
        """Constructs the reciprocal lattice dual to the real lattice

        Parameters
        ----------
        reallat : RealLattice
            Real Lattice

        Returns
        -------
        ReciprocalLatvec
            Reciprocal Lattice that is the dual of Real Lattice ``reallat``
        """
        tpiba = TPI / reallat.alat
        recvec = TPI * reallat.latvec_inv.T
        return cls(tpiba, recvec)

    @property
    def axes_tpiba(self) -> tuple[list[float], ...]:
        """List of vectors representing axes of lattice
        in units of 'tpiba'
        """
        return tuple(vec.tolist() for vec in self.recvec.T / self.tpiba)

    def cart2tpiba(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector coords in atomic units to 'tpiba' units"""
        return arr / self.tpiba

    def tpiba2cart(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector coords in 'tpiba' units to atomic units"""
        return arr * self.tpiba

    def cryst2tpiba(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector components in crystal coords to cartesian coords ('tpiba' units)"""
        return self.cryst2cart(arr, axis) / self.tpiba

    def tpiba2cryst(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector components in cartesian coords ('tpiba' units) to crystal coords"""
        return self.cart2cryst(arr, axis) * self.tpiba

    def dot(self, l_vec1: NDArray, l_vec2: NDArray, coords: str = 'cryst') -> NDArray:
        """Same as ``Lattice.dot()`` method, but exdending it for ``'tpiba'``.

        Parameters
        ----------
        l_vec1, l_vec2 : NDArray
            Input Array of vector components. Their first axes must be of length 3.
        coords : {'cryst', 'cart', 'tpiba'}; optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
            Array containing dot product between vectors in `l_vec1` and `l_vec2`
            of shape `(l_vec1.shape[1:], l_vec2.shape[1:])`.
        """
        if coords in ['cryst', 'cart']:
            return super().dot(l_vec1, l_vec2, coords)
        if coords == 'tpiba':
            return super().dot(l_vec1, l_vec2, 'cart') * self.tpiba**2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'tpiba'. Got {coords}")

    def norm2(self, l_vec: NDArray, coords: str = 'cryst') -> NDArray:
        """Same as `Lattice.dot()` method, but exdending it for ``'tpiba'``.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components. First axis must be of length 3.
        coords : {'cryst', 'cart', 'tpiba'}; optional
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
            Array containing norm squared of vectors given in `l_vec`.
        """
        if coords in ['cryst', 'cart']:
            return super().norm2(l_vec, coords)
        if coords == 'tpiba':
            return super().norm2(l_vec, 'cart') * self.tpiba ** 2
        else:
            raise ValueError(f"'coords' must be one of 'cryst', 'cart' or 'tpiba'. Got {coords}")
