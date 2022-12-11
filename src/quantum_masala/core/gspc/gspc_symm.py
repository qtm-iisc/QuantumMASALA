__all__ = ['SymmMod']

import numpy as np

from quantum_masala.core import Crystal
from quantum_masala.constants import TPIJ
from quantum_masala.core.spglib_utils import get_symmetry_crystal

ROUND_PREC: int = 6


class SymmMod:
    def __init__(self, crystal: Crystal, g_cryst: np.ndarray):
        recilat = crystal.recilat
        self.numg = g_cryst.shape[1]
        g_norm2 = recilat.norm2(g_cryst)
        idxsort = np.lexsort((*g_cryst, np.around(g_norm2, ROUND_PREC)))

        reallat_symm, recilat_symm = get_symmetry_crystal(crystal)
        if reallat_symm is not None:
            self.numsymm = len(reallat_symm)
            rot_recispc = recilat_symm
            trans_realspc = reallat_symm["translations"]
        else:
            self.numsymm = 1
            rot_recispc = np.eye(3).reshape(1, 3, 3)
            trans_realspc = np.zeros((1, 3))

        l_grot_rank = np.zeros((self.numsymm, self.numg), dtype='i8')
        for isymm in range(self.numsymm):
            grot_cryst = rot_recispc[isymm] @ g_cryst
            grot_norm2 = np.around(recilat.norm2(grot_cryst), ROUND_PREC)
            isort = np.lexsort((*grot_cryst, grot_norm2))
            l_grot_rank[isymm][isort] = np.arange(self.numg, dtype='i8')

        g_rank = np.amin(l_grot_rank, axis=0)
        _, l_istar = np.unique(g_rank, return_index=True)
        shell_idx = l_grot_rank.T[l_istar]
        shell_idx = idxsort[shell_idx]
        isort = np.argsort(shell_idx[:, 0])
        shell_idx = np.array(shell_idx[isort])

        l_grot_shell = np.tensordot(rot_recispc, g_cryst[:, shell_idx[:, 0]], axes=1)
        shell_phase = np.exp(TPIJ * np.sum(l_grot_shell * trans_realspc.reshape((-1, 3, 1)), axis=1))
        shell_phase = np.array(shell_phase.T)
        g_phase = np.zeros(self.numg, dtype='c16')
        g_count = np.zeros(self.numg, dtype='i8')
        for isymm in range(self.numsymm):
            g_phase[shell_idx[:, isymm]] += shell_phase[:, isymm]
            g_count[shell_idx[:, isymm]] += 1

        self.shell_idx = shell_idx
        self.shell_phase = shell_phase
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
