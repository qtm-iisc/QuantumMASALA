from __future__ import annotations
from typing import Optional
__all__ = ['BasisAtoms', 'PseudoPotFile']

import os
import numpy as np
from hashlib import md5
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qtm.lattice import RealLattice
from qtm.constants import ANGSTROM


class BasisAtoms:
    """Represents group of atoms of the same species in the unit cell of a
    crystal

    Parameters
    ----------
    label : str
        Label assigned to the atomic species
    mass : Optional[float]
        Atomic Mass of the species
    ppdata : PseudoPotFile
        Pseudo potential data of the species
    reallat : RealLattice
        Real Lattice of the crystal
    cryst : numpy.ndarray
        (``(3, numatoms)``) Crystal Coordinates of atoms in unit cell
    """

    def __init__(self, label: str, mass: Optional[float], ppdata: PseudoPotFile,
                 reallat: RealLattice,
                 cryst: np.ndarray, valence: float = None):
        self.label: str = label
        self.mass: Optional[float] = mass
        self.ppdata: PseudoPotFile = ppdata
        if ppdata is not None:
            self.valence: float = self.ppdata.valence
        elif valence is not None:
            self.valence: float = valence
        else:
            raise ValueError("'valence' needs to be specified when pseudopotential "
                             "data 'ppdata' is None. ")

        self.reallat: RealLattice = reallat
        if cryst.ndim != 2 or cryst.shape[0] != 3:
            raise ValueError("'cryst' must be a 2D array with `shape[0] == 3`. "
                             f"Got {cryst.shape}")
        self.cryst: np.ndarray = np.array(cryst, dtype='f8')
        self.numatoms: int = self.cryst.shape[1]

    @property
    def cart(self) -> np.ndarray:
        """numpy.ndarray: (``(3, self.numatoms)``) Cartesian Coords of the
        atoms in atomic units"""
        return self.reallat.cryst2cart(self.cryst)

    @property
    def alat(self) -> np.ndarray:
        """numpy.ndarray: (``(3, self.numatoms)``) Cartesian Coords of the
        atoms in 'alat' units"""
        return self.reallat.cryst2alat(self.cryst)

    @classmethod
    def from_cart(cls, label, mass, ppdata, reallat, *l_pos_cart):
        cart = np.array(l_pos_cart).T
        cryst = reallat.cart2cryst(cart)
        return cls(label, mass, ppdata, reallat, cryst)

    @classmethod
    def from_cryst(cls, label, mass, ppdata, reallat, *l_pos_cryst):
        cryst = np.array(l_pos_cryst).T
        return cls(label, mass, ppdata, reallat, cryst)

    @classmethod
    def from_alat(cls, label, mass, ppdata, reallat, *l_pos_alat, valence=None):
        alat = np.array(l_pos_alat).T
        cryst = reallat.alat2cryst(alat)
        return cls(label, mass, ppdata, reallat, cryst, valence)

    @classmethod
    def from_angstrom(cls, label, mass, ppdata, reallat, *l_pos_ang):
        cart = np.array(l_pos_ang).T * ANGSTROM
        cryst = reallat.cart2cryst(cart)
        return cls(label, mass, ppdata, reallat, cryst)

    @classmethod
    def from_bohr(cls, label, mass, ppdata, reallat, *l_pos_bohr):
        return cls.from_cart(label, mass, ppdata, reallat, *l_pos_bohr)


@dataclass
class PseudoPotFile(ABC):
    """Abstract Base class as template for Pseudopotential Reader.

    Pseudopotential readers will inherit this class and implement
    `read(dirname)` method for reading pseudopotential files.

    Attributes
    ----------
    dirname : str
        Path of the data file.
    filename : str
        Name of the file `os.path.basename`.
    md5_checksum :
        MD5 Hash of the data file.

    valence : int
        Number of valence electrons in the species according to pseudopotential.
    libxc_func: tuple[str, str]
        Pair of strings used to select the exchange and the correlation
        functionals respectively in the libxc library for computing
        XC Potentials.
    """
    dirname: str
    filename: str = field(init=False)
    md5_checksum: str = field(init=False)

    valence: int
    libxc_func: tuple[str, str]

    @classmethod
    @abstractmethod
    def from_file(cls, dirname: str, valence: int):
        return

    def __post_init__(self):
        self.filename = os.path.basename(self.dirname)

        hash_md5 = md5()
        with open(self.dirname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        self.md5_checksum = hash_md5.hexdigest()
