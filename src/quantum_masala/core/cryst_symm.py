__all__ = ['CrystalSymm']

import numpy as np
from spglib import get_symmetry

from quantum_masala import config


class CrystalSymm:

    def __init__(self, crystal: 'Crystal'):
        lattice = crystal.reallat.latvec.T
        positions = [sp.cryst.T for sp in crystal.l_atoms]
        numbers = np.repeat(range(len(positions)),
                            [len(pos) for pos in positions])
        positions = np.concatenate(positions, axis=0)
        reallat_symm = get_symmetry((lattice, positions, numbers),
                                    symprec=config.spglib_symprec)
        del reallat_symm['equivalent_atoms']
        if reallat_symm is None:
            reallat_symm = {
                'rotations': np.eye(3, dtype="i4").reshape((1, 3, 3)),
                'translations': np.zeros(3, dtype="f8"),
            }

        if config.symm_check_supercell:
            idx_identity = np.nonzero(
                np.all(reallat_symm['rotations'] - np.eye(3, dtype='i8'), axis=(1, 2))
            )[0]
            if len(idx_identity) != 1:
                idx_notrans = np.nonzero(
                    np.linalg.norm(reallat_symm['translations'], axis=1) <= config.spglib_symprec
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
        if config.symm_use_all_frac:
            return

        fac = np.multiply(self.symm['reallat_trans'], grid_shape)
        idx_comm = np.nonzero(
            np.linalg.norm(fac - np.rint(fac), axis=1) <= config.spglib_symprec
        )[0]
        self.symm = self.symm[idx_comm].copy()
        self.numsymm = len(self.symm)
