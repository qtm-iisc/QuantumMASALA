from __future__ import annotations
__all__ = ['Crystal', 'CrystalSymm']

import numpy as np
from spglib import get_symmetry

from qtm.config import qtmconfig
from qtm.lattice import RealLattice, ReciLattice
from qtm.crystal.basis_atoms import BasisAtoms

from qtm.msg_format import *


class Crystal:
    """Represents the structure of a Crystal in QuantumMASALA

    Parameters
    ----------
    reallat : RealLattice
        Represents the crystal's lattice in real space
    l_atoms : sequence of BasisAtoms
        Represents the crystal's atom basis where each element represents
        a subset of basis atoms belonging to the same species
    """
    def __init__(self, reallat: RealLattice, l_atoms: list[BasisAtoms]):
        if not isinstance(reallat, RealLattice):
            raise TypeError(type_mismatch_msg('reallat', reallat, RealLattice))
        self.reallat: RealLattice = reallat
        """Represents the crystal's lattice in real space"""
        self.recilat: ReciLattice = ReciLattice.from_reallat(self.reallat)
        """Represents the crystal's lattice in reciprocal space"""

        for ityp, typ in enumerate(l_atoms):
            if not isinstance(typ, BasisAtoms):
                raise TypeError(type_mismatch_msg(
                    f'l_atoms[{ityp}]', l_atoms[ityp], BasisAtoms
                ))
            if typ.reallat is not self.reallat:
                raise ValueError(obj_mismatch_msg(
                    f'l_atoms[{ityp}].reallat', typ.reallat,
                    'reallat', reallat
                ))
        self.l_atoms: list[BasisAtoms] = l_atoms
        """Represents the crystal's atom basis where each element represents
        a subset of basis atoms belonging to the same species"""
        self.symm: CrystalSymm = CrystalSymm(self)
        """Symmetry module of the Crystal"""

    @property
    def numel(self) -> int:
        """Total number of valence elecrons per unit cell in crystal"""
        return sum(sp.valence * sp.numatoms for sp in self.l_atoms)

    def gen_supercell(self, repeats: tuple[int, int, int]) -> Crystal:
        """Generates a supercell """
        try:
            repeats = tuple(repeats)
            for ni in repeats:
                if not isinstance(ni, int) or ni < 0:
                    raise TypeError
        except TypeError as e:
            raise TypeError(
                type_mismatch_seq_msg('repeats', repeats, 'positive integers')
            ) from e

        if len(repeats) != 3:
            raise ValueError("'repeats' must contain 3 elements. "
                             f"got {len(repeats)}")

        xi = [np.arange(n, dtype='i8') for n in repeats]
        grid = np.array(np.meshgrid(*xi, indexing='ij')).reshape((3, -1, 1))

        reallat = self.reallat
        alat_sup = repeats[0] * reallat.alat
        latvec_sup = repeats * reallat.latvec
        reallat_sup = RealLattice(alat_sup, latvec_sup)
        l_atoms_sup = []
        for sp in self.l_atoms:
            r_cryst = (grid + sp.r_cryst.reshape((3, 1, -1))).reshape(3, -1)
            r_cart_sup = reallat.cryst2cart(r_cryst)
            r_cryst_sup = reallat_sup.cart2cryst(r_cart_sup)
            l_atoms_sup.append(
                BasisAtoms(sp.label, sp.ppdata, sp.mass, reallat_sup, r_cryst_sup)
            )

        return Crystal(reallat_sup, l_atoms_sup)


class CrystalSymm:
    """Module for working with symmetries of given crystal"""

    symprec: float = 1E-5
    check_supercell: bool = True
    use_all_frac: bool = False

    def __init__(self, crystal: Crystal):
        assert isinstance(crystal, Crystal)

        lattice = crystal.reallat.latvec.T
        positions = [sp.r_cryst.T for sp in crystal.l_atoms]
        numbers = np.repeat(range(len(positions)),
                            [len(pos) for pos in positions])
        positions = np.concatenate(positions, axis=0)
        # print('type(lattice) :',type(lattice)) #debug statement
        # print('type(positions) :',type(positions)) #debug statement
        # print('type(numbers) :',type(numbers)) #debug statement
        if qtmconfig.gpu_enabled:
            reallat_symm = get_symmetry((lattice.get(), positions.get(), numbers),
                                    symprec=self.symprec)
        else:
            reallat_symm = get_symmetry((lattice, positions, numbers),
                                    symprec=self.symprec)
        del reallat_symm['equivalent_atoms']
        if reallat_symm is None:
            reallat_symm = {
                'rotations': np.eye(3, dtype="i4").reshape((1, 3, 3)),
                'translations': np.zeros(3, dtype="f8"),
            }

        if self.check_supercell:
            idx_identity = np.nonzero(
                np.all(reallat_symm['rotations'] == np.eye(3, dtype='i8'), axis=(1, 2))
            )[0]
            if len(idx_identity) != 1:
                idx_notrans = np.nonzero(
                    np.linalg.norm(reallat_symm['translations'], axis=1) <= self.symprec
                )[0]
                for k, v in reallat_symm.items():
                    reallat_symm[k] = v[idx_notrans]

        recilat_symm = np.linalg.inv(
            reallat_symm['rotations'].transpose((0, 2, 1))
        ).astype('i4')

        numsymm = len(reallat_symm['rotations'])
        self.symm: np.ndarray = np.array(
            [
                (reallat_symm['rotations'][i], reallat_symm['translations'][i],
                 recilat_symm[i]) for i in range(numsymm)
            ],
            dtype=[('reallat_rot', 'i4', (3, 3)), ('reallat_trans', 'f8', (3,)),
                   ('recilat_rot', 'i4', (3, 3))]
        )
        """List of Symmetry operations of input crystal"""

    @property
    def numsymm(self) -> int:
        """Total number of crystal symmetries"""
        return len(self.symm)

    @property
    def reallat_rot(self):
        return self.symm['reallat_rot']

    @property
    def reallat_trans(self):
        return self.symm['reallat_trans']

    @property
    def recilat_rot(self):
        return self.symm['recilat_rot']

    def filter_frac_trans(self, grid_shape: tuple[int, int, int]):
        if self.use_all_frac:
            return

        fac = np.multiply(self.symm['reallat_trans'], grid_shape)
        idx_comm = np.nonzero(
            np.linalg.norm(fac - np.rint(fac), axis=1) <= self.symprec
        )[0]
        self.symm = self.symm[idx_comm].copy()
