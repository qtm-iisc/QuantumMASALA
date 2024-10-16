from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
__all__ = ["Lattice", "RealLattice", "ReciLattice"]

import numpy as np

from qtm.config import NDArray, qtmconfig
from qtm.constants import TPI, ANGSTROM


class Lattice:
    """Represents the lattice of translations

    Describes a lattice by its primitive translation vectors and provides
    methods for coordinate transformations.

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
        if primvec.shape != (3, 3) or primvec.dtype != "f8":
            raise ValueError(
                "'primvec' must be a 2D array with shape (3, 3) and dtype 'f8'. got: "
                f"shape={primvec.shape if hasattr(primvec, 'shape') else 'NA'}, "
                f"dtype={primvec.dtype if hasattr(primvec, 'dtype') else 'NA'}"
            )

        if qtmconfig.gpu_enabled:
            import cupy

            primvec = cupy.asarray(primvec)

        self.primvec: NDArray = primvec.copy("C").astype("f8")
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
    def axes_cart(self) -> list[tuple[float, float, float]]:
        """tuple of the three primitive vectors in atomic units"""
        return list(tuple(vec.tolist()) for vec in self.primvec.T)

    def cart2cryst(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in cartesian coords
        to crystal coords.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in cartesian coords.
        axis : int, default=0
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

    def cryst2cart(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in crystal coords
        to cartesian coords.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in crystal coords.
        axis : int, default=0
            Axis indexing the vector coordinates/components.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as ``arr`` containing
            the vectors in cartesian coords.
        """
        assert arr.shape[axis] == 3, f"Expected arr.shape[{axis}] == 3, got {arr.shape}"
        vec_ = np.expand_dims(arr, axis)
        # FIXME: Double check the following line. In the old code, it was (3, 3) + (1,) * (arr.ndim - axis - 1)
        #        old code:
        #           mat_ = self.primvec.reshape((3, 3) + (1,) * (arr.ndim - axis - 1))
        #           return np.sum(mat_ * vec_, axis=axis + 1)
        mat_ = self.primvec.reshape((3, 3) + (1,) * (vec_.ndim - axis - 2))
        return np.sum(mat_ * vec_, axis=axis + 1)

    def dot(
        self,
        l_vec1: NDArray,
        l_vec2: NDArray,
        coords: Literal["cryst", "cart"] = "cryst",
    ) -> NDArray:
        """Computes the dot product between two sets of vectors

        Parameters
        ----------
        l_vec1, l_vec2 : NDArray
            Input Array of vector components.
            Their first axes must have length 3.
        coords : {'cryst', 'cart'}, default='cryst'
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
            raise ValueError(
                "Leading dimension of input arrays must be 3. "
                f"Got {l_vec1.shape}, {l_vec2.shape}"
            )

        shape_ = (*l_vec1.shape[1:], *l_vec2.shape[1:])
        l_vec1 = l_vec1.reshape(3, -1)
        l_vec2 = l_vec2.reshape(3, -1)
        if coords == "cryst":
            return (l_vec1.T @ self.metric @ l_vec2).reshape(shape_)
        elif coords == "cart":
            return (l_vec1.T @ l_vec2).reshape(shape_)
        else:
            raise ValueError(f"'coords' must be either 'cryst' or 'cart'. Got {coords}")

    def norm2(
        self, l_vec: NDArray, coords: Literal["cryst", "cart"] = "cryst"
    ) -> NDArray:
        """Computes the norm squared of input vectors.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart'}, default='cryst'
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
            raise ValueError(
                "Leading dimension of input array must be 3. " f"Got {l_vec.shape}"
            )
        l_vec = l_vec.reshape(3, -1)
        shape_ = l_vec.shape[1:]
        if coords == "cryst":
            return np.sum(l_vec * (self.metric @ l_vec), axis=0).reshape(shape_)
        elif coords == "cart":
            return np.sum(l_vec * l_vec, axis=0).reshape(shape_)
        else:
            raise ValueError(f"'coords' must be either 'cryst' or 'cart'. Got {coords}")

    def norm(
        self, l_vec: NDArray, coords: Literal["cryst", "cart"] = "cryst"
    ) -> NDArray:
        """Computes the norm of given vectors.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart'}, default='cryst'
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

    def __eq__(self, other: Lattice):
        if type(self) is not type(other):
            return False
        return (
            np.linalg.norm(self.primvec - np.asarray(other.primvec, like=self.primvec))
            <= 1e-5
        )
    
    def __repr__(self) -> str:
        return f"Lattice(primvec={self.primvec})"
    

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
        if qtmconfig.gpu_enabled:
            import cupy

            self.latvec = cupy.asarray(self.latvec)
            # print("type(self.latvec) :", type(self.latvec))  # debug statement
        self.latvec_inv: NDArray = self.primvec_inv
        """(``(3, 3)``, ```'f8'``) Alias of ``primvec_inv``
        """
        self.adot: NDArray = self.metric
        """(``(3, 3)``, ```'f8'``) Alias of ``metric``
        """

    @classmethod
    def from_bohr(cls, alat: float, a1, a2, a3) -> RealLattice:
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
    def from_angstrom(cls, alat: float, a1, a2, a3) -> RealLattice:
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
    def from_alat(cls, alat: float, a1, a2, a3) -> RealLattice:
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
    def from_recilat(cls, recilat: "ReciLattice") -> RealLattice:
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
    def axes_alat(self) -> list[tuple[float, float, float]]:
        """List of the three primitive vectors in 'alat' units"""
        return list(tuple(vec) for vec in self.latvec.T / self.alat)

    def cart2alat(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Transforms array of vector components in atomic units
        to 'alat' units.

        Parameters
        ----------
        arr : NDArray
            Input array of vector components in atomic units.
        axis : int, default=0
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
        axis : int, default=0
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
        axis : int, default=0
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
        axis : int, default=0
            Axis indexing the ``i``th coordinate in ``arr``.
            ``arr.shape[axis]`` must be equal to 3.

        Returns
        -------
        NDArray
            Array with the same shape as ``arr`` containing
            the vectors in crystal coords.
        """
        return self.cart2cryst(arr, axis) * self.alat

    def dot(
        self,
        l_vec1: NDArray,
        l_vec2: NDArray,
        coords: Literal["cryst", "cart", "alat"] = "cryst",
    ) -> NDArray:
        """Same as ``Lattice.dot()`` method, but exdending it for
        ``coords='alat'``.

        Parameters
        ----------
        l_vec1, l_vec2 : NDArray
            Input Array of vector components.
            Their first axes must have length 3.
        coords : {'cryst', 'cart', 'alat'}, defualt='cryst'
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
        if coords in ["cryst", "cart"]:
            return super().dot(l_vec1, l_vec2, coords)
        if coords == "alat":
            return super().dot(l_vec1, l_vec2, "cart") * self.alat**2
        else:
            raise ValueError(
                f"'coords' must be one of 'cryst', 'cart' or 'alat'. Got {coords}"
            )

    def norm2(
        self, l_vec: NDArray, coords: Literal["cryst", "cart", "alat"] = "cryst"
    ) -> NDArray:
        """Same as ``Lattice.norm2()`` method, but exdending it for
        ``coords='alat'``.

        Parameters
        ----------
        l_vec : NDArray
            (``(3, ...)``) Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart', 'alat'}, default='cryst'
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
        if coords in ["cryst", "cart"]:
            return super().norm2(l_vec, coords)
        if coords == "alat":
            return super().norm2(l_vec, "cart") * self.alat**2
        else:
            raise ValueError(
                "'coords' must be one of 'cryst', 'cart' or 'alat'. " f"Got {coords}"
            )

    def norm(
        self, l_vec: NDArray, coords: Literal["cryst", "cart", "alat"] = "cryst"
    ) -> NDArray:
        """Computes the norm of given vectors.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components.
            First axis must be of length 3.
        coords : {'cryst', 'cart', 'alat'}, default='cryst'
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
        if coords in ["cryst", "cart"]:
            return super().norm(l_vec, coords)
        if coords == "alat":
            return super().norm(l_vec, "cart") * self.alat
        else:
            raise ValueError(
                "'coords' must be one of 'cryst', 'cart' or 'alat'. " f"Got {coords}"
            )

    def get_mesh_coords(
        self,
        n1: int,
        n2: int,
        n3: int,
        coords: str = "cryst",
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """ Generates a mesh of coordinates in the unit cell.

            Parameters
            ----------
            n1 : int
                The number of points along the first lattice vector.
            n2 : int
                The number of points along the second lattice vector.
            n3 : int
                The number of points along the third lattice vector.
            coords : str, optional
                The coordinate system to generate the mesh in. Possible values are "cryst" (default), "cart", and "alat".
            origin : tuple[float, float, float], optional
                The origin of the mesh in the specified coordinate system. Defaults to (0.0, 0.0, 0.0).

            Returns
            -------
            numpy.ndarray
                The mesh of coordinates in the specified coordinate system.

            Raises
            ------
            ValueError
                If any of the n1, n2, n3 values are not positive integers.
        """
        xi = []
        for i, n in enumerate([n1, n2, n3]):
            if not isinstance(n, int) or n < 1:
                raise ValueError(
                    f"'n{i+1}' must be a positive integer. " f"got {n} (type {type(n)})"
                )
            xi.append(np.arange(n) / n)
        r_cryst = np.array(np.meshgrid(*xi, indexing="ij"))

        if coords == "cart":
            origin = self.cart2cryst(origin)
        elif coords == "alat":
            origin = self.alat2cryst(origin)
        r_cryst -= origin.reshape((3, 1, 1, 1))
        r_cryst -= np.rint(r_cryst)

        if coords == "cryst":
            return r_cryst
        elif coords == "cart":
            return self.cryst2cart(r_cryst)
        elif coords == "alat":
            return self.cryst2alat(r_cryst)
        
    def __repr__(self, indent="        ") -> str:
        # latvec = "\n".join(f"{indent}  {vec}" for vec in self.latvec)
        latvec = f"{indent}    "+str(self.latvec).replace("\n", f"\n{indent}    ")
        return f"RealLattice(\n{indent}  alat={self.alat}, \n{indent}  latvec=\n{latvec},\n{indent})"
    
    def __str__(self) -> str:
        alat_str = f"Lattice parameter 'alat' :   {self.alat:.5f}  a.u."
        cellvol_str = f"Unit cell volume         :  {self.cellvol:.5f}  (a.u.)^3"
        num_atoms_str = "Number of atoms/cell     : 1"
        num_types_str = "Number of atomic types   : 1"
        num_electrons_str = "Number of electrons      : 16"

        crystal_axes_str = "Crystal Axes: coordinates in units of 'alat' ({:.5f} a.u.)".format(self.alat)
        crystal_axes = "\n".join(
            f"    a({i+1}) = ({vec[0]:8.5f}, {vec[1]:8.5f}, {vec[2]:8.5f})"
            for i, vec in enumerate(self.latvec.T/ self.alat)
        )

        reci_lattice = ReciLattice.from_reallat(self)
        tpiba_str = "Reciprocal Axes: coordinates in units of 'tpiba' ({:.5f} (a.u.)^-1)".format(reci_lattice.tpiba)
        reci_axes = "\n".join(
            f"    b({i+1}) = ({vec[0]:8.5f}, {vec[1]:8.5f}, {vec[2]:8.5f})"
            for i, vec in enumerate(reci_lattice.recvec.T/ reci_lattice.tpiba)
        )

        return (
            f"{alat_str}\n"
            f"{cellvol_str}\n"
            f"{num_atoms_str}\n"
            f"{num_types_str}\n"
            f"{num_electrons_str}\n\n"
            f"{crystal_axes_str}\n"
            f"{crystal_axes}\n\n"
            f"{tpiba_str}\n"
            f"{reci_axes}"
        )

        


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
    def from_cart(cls, tpiba: float, b1, b2, b3) -> ReciLattice:
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
    def from_tpiba(cls, tpiba: float, b1, b2, b3) -> ReciLattice:
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
    def from_reallat(cls, reallat: RealLattice) -> ReciLattice:
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
    def axes_tpiba(self) -> list[tuple[float, float, float]]:
        """List of vectors representing axes of lattice
        in units of 'tpiba'
        """
        return list(tuple(vec) for vec in self.recvec.T / self.tpiba)

    def cart2tpiba(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector coords in atomic units to 'tpiba' units"""
        return arr / self.tpiba

    def tpiba2cart(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector coords in 'tpiba' units to atomic units"""
        return arr * self.tpiba

    def cryst2tpiba(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector components in crystal coords to
        cartesian coords ('tpiba' units)"""
        return self.cryst2cart(arr, axis) / self.tpiba

    def tpiba2cryst(self, arr: NDArray, axis: int = 0) -> NDArray:
        """Converts array of vector components in cartesian coords
        ('tpiba' units) to crystal coords"""
        return self.cart2cryst(arr, axis) * self.tpiba

    def dot(
        self,
        l_vec1: NDArray,
        l_vec2: NDArray,
        coords: Literal["cryst", "cart", "alat"] = "cryst",
    ) -> NDArray:
        """Same as ``Lattice.dot()`` method, but exdending it for ``'tpiba'``.

        Parameters
        ----------
        l_vec1, l_vec2 : NDArray
            Input Array of vector components. Their first axes must be of length 3.
        coords : {'cryst', 'cart', 'tpiba'}; default='cryst'
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
            Array containing dot product between vectors in `l_vec1` and `l_vec2`
            of shape `(l_vec1.shape[1:], l_vec2.shape[1:])`.
        """
        if coords in ["cryst", "cart"]:
            return super().dot(l_vec1, l_vec2, coords)
        if coords == "tpiba":
            return super().dot(l_vec1, l_vec2, "cart") * self.tpiba**2
        else:
            raise ValueError(
                f"'coords' must be one of 'cryst', 'cart' or 'tpiba'. Got {coords}"
            )

    def norm2(
        self, l_vec: NDArray, coords: Literal["cryst", "cart", "alat"] = "cryst"
    ) -> NDArray:
        """Same as `Lattice.dot()` method, but exdending it for ``'tpiba'``.

        Parameters
        ----------
        l_vec : NDArray
            Input Array of vector components. First axis must be of length 3.
        coords : {'cryst', 'cart', 'tpiba'}, default='cryst'
            Coordinate type of input vector components.

        Returns
        -------
        NDArray
            Array containing norm squared of vectors given in `l_vec`.
        """
        if coords in ["cryst", "cart"]:
            return super().norm2(l_vec, coords)
        if coords == "tpiba":
            return super().norm2(l_vec, "cart") * self.tpiba**2
        else:
            raise ValueError(
                f"'coords' must be one of 'cryst', 'cart' or 'tpiba'. Got {coords}"
            )

    def __repr__(self) -> str:
        return f"ReciLattice(tpiba={self.tpiba}, recvec={self.recvec})"