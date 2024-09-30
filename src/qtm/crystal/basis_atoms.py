from __future__ import annotations
__all__ = ['BasisAtoms', 'PseudoPotFile']

import os
import numpy as np
from hashlib import md5
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qtm.lattice import RealLattice

from qtm.config import NDArray
from qtm.msg_format import *

from qtm.constants import ANGSTROM


@dataclass
class PseudoPotFile(ABC):
    """Template for Pseudopotential Data container with attached reader
    routines.

    Pseudopotential data containers will inherit this class and implement
    `from_file` method for reading pseudopotential files.
    """

    dirname: str
    """Path of the data file."""
    filename: str = field(init=False)
    """ Name of the file `os.path.basename`."""
    md5_checksum: str = field(init=False)
    """MD5 Hash of the data file."""
    valence: int
    """Number of valence electrons per atom according to pseudopotential"""
    libxc_func: tuple[str, str]
    """Pair of strings used to select the exchange and the correlation functionals
    respectively in the libxc library for computing XC Potentials"""

    @classmethod
    @abstractmethod
    def from_file(cls, dirname: str) -> PseudoPotFile:
        """Method to parse input files and generate `PseudoPotFile` instances
        containing its data.

        Parameters
        ----------
        dirname : str
            Path of the input file.
        """
        pass

    def __post_init__(self):
        self.filename = os.path.basename(self.dirname)

        hash_md5 = md5()
        with open(self.dirname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        self.md5_checksum = hash_md5.hexdigest()
    
    def __str__(self, indent="") -> str:
        return f"PseudoPotFile(\n{indent}  dirname='{self.dirname}', \n{indent}  valence={self.valence}, \n{indent}  md5_checksum='{self.md5_checksum}')"

    def __repr__(self, indent="") -> str:
        return self.__str__()

class BasisAtoms:
    """Represents group of atoms of the same species in the unit cell of a
    crystal

    Parameters
    ----------
    label : str
        Label assigned to the atomic species
    ppdata : PseudoPotFile | int
        Pseudopotential data of the species. For routines where pseudopotentials
        are not involed the number of valence electrons of input species can
        be given instead.
    mass : float | None, default=None
        Atomic Mass of the species. If not required, can be given None.
    reallat : RealLattice
        Real Lattice of the crystal.
    r_cryst : NDArray
        (``(3, numatoms)``) Crystal Coordinates of atoms in unit cell.
    """

    def __init__(self, label: str, ppdata: PseudoPotFile | int, mass: float | None,
                 reallat: RealLattice, r_cryst: NDArray,
                 ):
        self.label: str = label
        """Label assigned to the atomic species."""

        if not isinstance(reallat, RealLattice):
            raise TypeError(type_mismatch_msg('reallat', reallat, RealLattice))
        self.reallat: RealLattice = reallat
        """Real Lattice of the crystal"""

        if isinstance(ppdata, int):
            valence = ppdata
            ppdata = None
        elif isinstance(ppdata, PseudoPotFile):
            valence = ppdata.valence
        elif ppdata is None:
            valence = -1
        else:
            raise TypeError(type_mismatch_msg('ppdata', ppdata, [PseudoPotFile, int]))
        self.valence: int = valence
        """Number of valence electrons per atom."""
        self.ppdata: PseudoPotFile | None = ppdata
        """ Pseudopotential data of the species."""

        if mass is not None:
            if not isinstance(mass, float) or mass < 0:
                raise TypeError(type_mismatch_msg('mass', mass, "a positive float"))
        self.mass: float | None = mass
        """Mass of the atomic species in a.m.u."""

        try:
            r_cryst = np.asarray(r_cryst, dtype='f8', order='C',
                                 like=self.reallat.latvec)
        except Exception as e:
            raise TypeError(type_mismatch_msg('r_cryst', r_cryst, NDArray)) from e
        if r_cryst.ndim != 2:
            raise ValueError(value_mismatch_msg('r_cryst.ndim', r_cryst.ndim, 2))
        if r_cryst.shape[0] != 3:
            raise ValueError(
                value_mismatch_msg('r_cryst.shape[0]', r_cryst.shape[0], 3)
            )
        self.r_cryst: NDArray = r_cryst
        """(``(3, self.numatoms)``) Crystal Coordinates of atoms in unit cell"""

    @property
    def numatoms(self) -> int:
        """Number of atoms per unit cell belonging to the species"""
        return self.r_cryst.shape[1]

    @property
    def r_cart(self) -> NDArray:
        """(``(3, self.numatoms)``) Cartesian Coords of the
        atoms in atomic units"""
        return self.reallat.cryst2cart(self.r_cryst)

    @property
    def r_alat(self) -> NDArray:
        """(``(3, self.numatoms)``) Cartesian Coords of the
        atoms in 'alat' units"""
        return self.reallat.cryst2alat(self.r_cryst)

    @classmethod
    def from_cart(cls, label: str, ppdata: PseudoPotFile | int | None, mass: float | None,
                  reallat: RealLattice, *r_cart) -> BasisAtoms:
        """Generates `BasisAtoms` instance from cartesian coordinates

        Parameters
        ----------
        label : str
            Label assigned to the atomic species
        ppdata : PseudoPotFile | int
            Pseudopotential data of the species. Alternatively, the number of
            valence electrons of input species can be given instead for routines
            where Pseudopotentials are not involved.
        reallat : RealLattice
            Real Lattice of the crystal
        *r_cart : tuple of coordinates
            Cartesian Coordinates of atoms in unit cell, each given by a
            3-element sequence of numbers.
        mass : float | None, default=None
            Atomic Mass of the species
        """
        if not isinstance(reallat, RealLattice):
            raise TypeError(type_mismatch_msg('reallat', reallat, RealLattice))
        r_cart = np.array(r_cart, dtype='f8', like=reallat.latvec).T
        r_cryst = reallat.cart2cryst(r_cart)
        return cls(label, ppdata, mass, reallat, r_cryst)

    @classmethod
    def from_cryst(cls, label: str, ppdata: PseudoPotFile | int, mass: float | None,
                   reallat: RealLattice, *r_cryst) -> BasisAtoms:
        """Generates `BasisAtoms` instance from crystal coordinates.
        Refer to `from_cart` for a descripton of the input
        arguments"""
        if not isinstance(reallat, RealLattice):
            raise TypeError(type_mismatch_msg('reallat', reallat, RealLattice))
        r_cryst = np.array(r_cryst, dtype='f8', like=reallat.latvec).T
        return cls(label, ppdata, mass, reallat, r_cryst)

    @classmethod
    def from_alat(cls, label: str, ppdata: PseudoPotFile | int, mass: float | None,
                  reallat: RealLattice, r_alat) -> BasisAtoms:
        """Generates `BasisAtoms` instance from cartesian coordinates in
        alat units. Refer to `from_cart` for a descripton of the input
        arguments"""
        if not isinstance(reallat, RealLattice):
            raise TypeError(type_mismatch_msg('reallat', reallat, RealLattice))
        r_alat = np.array(r_alat, dtype='f8', like=reallat.latvec).T
        r_cryst = reallat.alat2cryst(r_alat)
        return cls(label, ppdata, mass, reallat, r_cryst)

    @classmethod
    def from_angstrom(cls, label: str, ppdata: PseudoPotFile | int, mass: float | None,
                      reallat: RealLattice, *r_ang) -> BasisAtoms:
        """Generates `BasisAtoms` instance from cartesian coordinates in
        angstrom units. Refer to `from_cart` for a descripton of the input
        arguments"""
        if not isinstance(reallat, RealLattice):
            raise TypeError(type_mismatch_msg('reallat', reallat, RealLattice))
        r_cart = np.array(r_ang, dtype='f8', like=reallat.latvec).T * ANGSTROM
        r_cryst = reallat.cart2cryst(r_cart)
        return cls(label, ppdata, mass, reallat, r_cryst)

    @classmethod
    def from_bohr(cls, label: str, ppdata: PseudoPotFile | int, mass: float | None,
                  reallat: RealLattice, *r_cart) -> BasisAtoms:
        """Alias of `from_cart` classmethod"""
        return cls.from_cart(label, ppdata, mass, reallat, *r_cart)


    def __repr__(self, indent="") -> str:
        r_cryst_str = ""
        for i in range(self.numatoms):
            r_cryst_str += f"\n{indent}    {np.array2string(self.r_cryst[:, i], separator=', ')},"
        res = f"{indent}BasisAtoms(\n{indent}  label='{self.label}', \n{indent}  ppdata={self.ppdata.__str__(indent+'  ')}, \n{indent}  mass={self.mass}, \n{indent}  r_cryst=({r_cryst_str}))"
        return res