

__all__ = ["AtomBasis", "Crystal"]

from typing import Optional
import numpy as np
from quantum_masala.core import RealLattice, ReciprocalLattice, PseudoPotFile


class AtomBasis:
    """Represents group of atoms of the same species in the unit cell of a
    crystal

    Parameters
    ----------
    label : str
        Label assigned to the atomic species
    mass : Optional[float]
        Atomic Mass of the species
    ppdata
        Pseudo potential data of the species
    reallat : RealLattice
        Real Lattice of the crystal
    cryst : numpy.ndarray
        (``(3, numatoms)``) Crystal Coordinates of atoms in unit cell
    """

    def __init__(self, label: str, mass: Optional[float], ppdata, reallat: RealLattice,
                 cryst: np.ndarray):
        self.label: str = label
        self.mass: Optional[float] = mass
        self.ppdata: PseudoPotFile = ppdata

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
    def from_alat(cls, label, mass, ppdata, reallat, *l_pos_alat):
        alat = np.array(l_pos_alat).T
        cryst = reallat.alat2cryst(alat)
        return cls(label, mass, ppdata, reallat, cryst)


class Crystal:

    def __init__(self, reallat, l_atoms: list[AtomBasis]):
        self.reallat = reallat
        self.recilat = ReciprocalLattice.from_reallat(self.reallat)

        self.l_atoms = l_atoms

        lattice = self.reallat.latvec.T
        positions = [sp.cryst.T for sp in self.l_atoms]
        numbers = sum(((isp,) * len(pos) for isp, pos in enumerate(positions)),
                      ())
        positions = np.concatenate(positions, axis=1)
        self.spglib_cell = (lattice, positions, numbers)

    @property
    def numel(self) -> int:
        numel = 0
        for sp in self.l_atoms:
            if sp.ppdata is None:
                raise ValueError("cannot compute 'numel' without 'ppdata' "
                                 "specified for all atoms")
            numel += sp.ppdata.valence * sp.numatoms
        return round(numel)
