__all__ = ['SymmMod']

import numpy as np

from quantum_masala import pw_logger
from quantum_masala.core import Crystal
from quantum_masala.constants import TPIJ

ROUND_PREC: int = 6


class SymmMod:
    @pw_logger.time('gspc_symm:init')
    def __init__(self, crystal: Crystal, gspc: 'GSpace'):
        self.numg = gspc.numg
        grid_shape = gspc.grid_shape
        nx, ny, nz = grid_shape
        g_cryst = gspc.cryst
        g_norm2 = gspc.norm2

        crystal_symm = crystal.get_symmetry(grid_shape=gspc.grid_shape)
        self.numsymm = len(crystal_symm)
        recilat_rot = crystal_symm['recilat_rot']
        reallat_trans = crystal_symm['reallat_trans']

        idxsort = np.arange(len(g_norm2), dtype='i8')
        gsort_norm2 = g_norm2[idxsort]
        _, isplit = np.unique(np.around(gsort_norm2, ROUND_PREC), return_index=True)
        l_shellidx = np.split(idxsort, isplit)[1:]
        l_groupidx = []
        l_groupphase = []
        for shellidx in l_shellidx:
            gshell_cryst = g_cryst[:, shellidx]

            grot_cryst = np.tensordot(recilat_rot, gshell_cryst, axes=1)
            ix, iy, iz = [grot_cryst[:, i] + grid_shape[i] // 2 for i in range(3)]
            grot_rank = iz + (nz * iy) + (nz * ny * ix)
            gshell_rank = grot_rank[0]

            oob = (ix < 0) + (iy < 0) + (iz < 0) \
                + (ix >= nx) + (iy >= ny) + (iz >= nz)
            if np.any(oob):
                raise ValueError("Rotated vectors do not map to original vectors "
                                 "due to insufficient grid shape")

            gstar_rank = np.amin(grot_rank, axis=0)
            _, gstar_idx = np.unique(gstar_rank, return_index=True)
            groupidx = np.searchsorted(gshell_rank, grot_rank[:, gstar_idx])

            l_groupidx.append(shellidx[groupidx.T])
            groupphase = np.exp(
                TPIJ * np.sum(reallat_trans.reshape(-1, 3, 1)
                              * grot_cryst[:, :, gstar_idx], axis=1)
            ).T
            l_groupphase.append(groupphase)

        self.shell_idx = np.concatenate(l_groupidx, axis=0)
        self.shell_phase = np.concatenate(l_groupphase, axis=0)

        g_phase = np.zeros(self.numg, dtype='c16')
        g_count = np.zeros(self.numg, dtype='i8')
        for isymm in range(self.numsymm):
            idxg = self.shell_idx[:, isymm]
            g_phase[idxg] += self.shell_phase[:, isymm]
            g_count[idxg] += 1
        self.g_phaseconj = np.conj(g_phase) / g_count

    def _symmetrize(self, arr_g: np.ndarray):
        shell_avg = np.empty(self.numg, dtype="c16")
        shell_idx_ = (Ellipsis, self.shell_idx)
        shell_avg[shell_idx_] = (
            np.sum(arr_g[shell_idx_] * self.shell_phase, axis=-1, keepdims=True)
            / self.numsymm
        )
        shell_avg *= self.g_phaseconj

        return shell_avg

    def symmetrize(self, arr_g: np.ndarray):
        if arr_g.shape[-1] != self.numg:
            raise ValueError(
                f"'arr_g.shape' invalid. Expected {(..., self.numg)}, got {arr_g.shape}"
            )

        arr_ = arr_g.reshape(-1, self.numg)
        for i in range(arr_.shape[0]):
            arr_[i] = self._symmetrize(arr_[i])

        return arr_g
