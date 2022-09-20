import numpy as np

from ..cryst import Crystal
from .utils import get_symmetry_crystal
from ..gspc import GSpace

from ..constants import TPIJ


class SymmGspc:
    def __init__(self, crystal: Crystal, gspc: GSpace):
        reallat_symm, recilat_symm = get_symmetry_crystal(crystal)

        if reallat_symm is not None:
            self.numsymm = len(reallat_symm)
            rot_recispc = recilat_symm
            trans_realspc = reallat_symm["translations"]
        else:
            self.numsymm = 1
            rot_recispc = np.eye(3).reshape(1, 3, 3)
            trans_realspc = np.zeros((1, 3))

        self.gspc_size = gspc.numg

        ix, iy, iz = gspc.cryst
        ax, ay, az = np.amin(ix), np.amin(iy), np.amin(iz)
        bx, by, bz = np.amax(ix) - ax + 1, np.amax(iy) - ay + 1, np.amax(iz) - az + 1

        rank = (ix - ax) + bx * (iy - ay) + bx * by * (iz - az) + 1

        rank2idx = np.ones(bx * by * bz, dtype="i8") * -1
        rank2idx[rank] = np.arange(self.gspc_size)

        l_cryst_rot = np.tensordot(rot_recispc, gspc.cryst, axes=1)
        l_phase_rot = np.exp(
            TPIJ
            * np.sum(l_cryst_rot * trans_realspc.reshape((self.numsymm, 3, 1)), axis=1)
        )
        l_phase_rot = np.ascontiguousarray(l_phase_rot.T)

        l_rank_rot = np.rint(
            (l_cryst_rot[:, 0] - ax)
            + bx * (l_cryst_rot[:, 1] - ay)
            + bx * by * (l_cryst_rot[:, 2] - az)
            + 1
        ).astype("i8")

        l_rank_rot = np.ascontiguousarray(l_rank_rot.T)

        g_done = np.zeros(self.gspc_size, dtype=bool)
        g_phase = np.zeros(self.gspc_size, dtype="c16")
        g_count = np.zeros(self.gspc_size, dtype="i8")

        shell_phase = []
        shell_idx = []

        for ig in range(self.gspc_size):
            if g_done[ig]:
                continue

            g_rank_rot = l_rank_rot[ig]
            g_phase_rot = l_phase_rot[ig]
            ig_rot = rank2idx[g_rank_rot]

            for isymm in range(self.numsymm):
                jg = ig_rot[isymm]
                g_phase[jg] += g_phase_rot[isymm]
                g_count[jg] += 1

            shell_idx.append(ig_rot)
            shell_phase.append(g_phase_rot)
            g_done[ig_rot] = True

        self.shell_phase = np.array(shell_phase)
        self.g_phaseconj = np.conj(g_phase) / g_count
        self.shell_idx = np.array(shell_idx)

    def _symmetrize(self, arr_g: np.ndarray):
        shell_avg = np.empty(self.gspc_size, dtype="c16")
        shell_idx_ = (Ellipsis, self.shell_idx)
        shell_avg[shell_idx_] = (
            np.sum(arr_g[shell_idx_] * self.shell_phase, axis=-1, keepdims=True)
            / self.numsymm
        )
        shell_avg *= self.g_phaseconj

        return shell_avg

    def symmetrize(self, arr_g: np.ndarray):
        if arr_g.shape[-1] != self.gspc_size:
            raise ValueError(
                f"'arr_g.shape' invalid. Expected {(..., self.gspc_size)}, got {arr_g.shape}"
            )

        arr_shape = arr_g.shape
        arr_g = arr_g.reshape(-1, self.gspc_size)
        for i in range(arr_g.shape[0]):
            arr_g[i] = self._symmetrize(arr_g[i])

        return arr_g.reshape(arr_shape)
