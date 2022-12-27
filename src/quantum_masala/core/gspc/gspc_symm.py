__all__ = ['SymmMod']

import numpy as np

from quantum_masala import pw_counter
from quantum_masala.core import Crystal
from quantum_masala.constants import TPIJ
from quantum_masala.core.spglib_utils import get_symmetry_crystal

ROUND_PREC: int = 6


class SymmMod:
    def __init__(self, crystal: Crystal, gspc: 'GSpace'):
        pw_counter.start_clock('gspc_symm:init')
        self.numg = gspc.numg
        grid_shape = gspc.grid_shape
        nx, ny, nz = grid_shape
        g_cryst = gspc.cryst
        g_norm2 = gspc.norm2

        reallat_symm, recilat_symm = get_symmetry_crystal(crystal)
        if reallat_symm is not None:
            self.numsymm = len(reallat_symm)
            rot_recispc = recilat_symm
            trans_realspc = reallat_symm["translations"]
        else:
            self.numsymm = 1
            rot_recispc = np.eye(3).reshape(1, 3, 3)
            trans_realspc = np.zeros((1, 3))

        idxsort = np.arange(len(g_norm2), dtype='i8')
        gsort_norm2 = g_norm2[idxsort]
        _, isplit = np.unique(np.around(gsort_norm2, ROUND_PREC), return_index=True)
        l_shellidx = np.split(idxsort, isplit)[1:]
        l_groupidx = []
        l_groupphase = []
        for shellidx in l_shellidx:
            gshell_cryst = g_cryst[:, shellidx]

            grot_cryst = np.tensordot(rot_recispc, gshell_cryst, axes=1)
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
                TPIJ * np.sum(trans_realspc.reshape(-1, 3, 1)
                              * grot_cryst[:, :, gstar_idx], axis=1)
            ).T
            l_groupphase.append(groupphase)

        pw_counter.stop_clock('gspc_symm:init')

        self.shell_idx = np.concatenate(l_groupidx, axis=0)
        self.shell_phase = np.concatenate(l_groupphase, axis=0)
        print(self.numsymm)
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
