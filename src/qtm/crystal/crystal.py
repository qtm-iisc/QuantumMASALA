# from __future__ import annotations
__all__ = ['Crystal', 'CrystalSymm']

import numpy as np
from spglib import get_symmetry

from qtm.lattice import RealLattice, ReciLattice
from qtm.crystal.basis_atoms import BasisAtoms


class Crystal:

    def __init__(self, reallat, l_atoms: list[BasisAtoms]):
        self.reallat: RealLattice = reallat
        self.recilat: ReciLattice = ReciLattice.from_reallat(self.reallat)

        self.l_atoms: list[BasisAtoms] = l_atoms
        self.symm: CrystalSymm = CrystalSymm(self)

    @property
    def numel(self) -> int:
        numel = 0
        for sp in self.l_atoms:
            if sp.ppdata is None:
                raise ValueError("cannot compute 'numel' without 'ppdata' "
                                 "specified for all atoms")
            numel += sp.ppdata.valence * sp.numatoms
        return round(numel)

    def gen_supercell(self, repeats: tuple[int, int, int]):
        repeats = tuple(repeats)
        if len(repeats) != 3:
            raise ValueError("length of 'repeats' must be 3. "
                             f"got {len(repeats)}")
        for i, ni in enumerate(repeats):
            if not isinstance(ni, int) or ni < 1:
                raise ValueError(f"'repeats' must be a tuple of 3 positive integers. "
                                 f"Got repeats[{i}] = {ni} (type {type(ni)})")

        xi = [np.arange(n, dtype='i8') for n in repeats]
        grid = np.array(np.meshgrid(*xi, indexing='ij')).reshape((3, -1, 1))

        reallat = self.reallat
        alat_sup = repeats[0] * reallat.alat
        latvec_sup = repeats * reallat.latvec
        reallat_sup = RealLattice(alat_sup, latvec_sup)
        l_atoms_sup = []
        for sp in self.l_atoms:
            cryst = (grid + sp.cryst.reshape((3, 1, -1))).reshape(3, -1)
            cart_sup = reallat.cryst2cart(cryst)
            cryst_sup = reallat_sup.cart2cryst(cart_sup)
            l_atoms_sup.append(
                BasisAtoms(sp.label, sp.mass, sp.ppdata, reallat_sup, cryst_sup)
            )

        return Crystal(reallat_sup, l_atoms_sup)


class CrystalSymm:

    symprec: float = 1E-5
    check_supercell: bool = True
    symm_check_supercell: bool = True
    use_all_frac: bool = False

    def __init__(self, crystal: Crystal):
        lattice = crystal.reallat.latvec.T
        positions = [sp.cryst.T for sp in crystal.l_atoms]
        numbers = np.repeat(range(len(positions)),
                            [len(pos) for pos in positions])
        positions = np.concatenate(positions, axis=0)
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
        self.symm = np.array(
            [
                (reallat_symm['rotations'][i], reallat_symm['translations'][i],
                 recilat_symm[i]) for i in range(numsymm)
            ],
            dtype=[('reallat_rot', 'i4', (3, 3)), ('reallat_trans', 'f8', (3,)),
                   ('recilat_rot', 'i4', (3, 3))]
        )
        self.numsymm = len(self.symm)

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
        self.numsymm = len(self.symm)
