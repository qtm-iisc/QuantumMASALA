import cupy as cp
from cupy.cublas import gemm

from quantum_masala.core import GkSpace, RField
from quantum_masala.pseudo import NonlocGenerator
from quantum_masala.dft import KSHam
from .fftmod_gpu import CpFFT3D


class KSHamGPU(KSHam):
    __slots__ = ["gkspc", "fft_mod", "ke_gk", "l_vkb", "l_vkb_H", "dij",
                 "noncolin", "numspin", "vloc_r", "idxspin"]

    def __init__(self, gkspc: GkSpace, is_spin: bool, is_noncolin: bool,
                 vloc: RField, l_nloc: list[NonlocGenerator]):
        super().__init__(gkspc, is_spin, is_noncolin, vloc, l_nloc)

        self.ke_gk = cp.asarray(self.ke_gk)
        for i, vkb_dij in enumerate(self.l_vkb_dij):
            self.l_vkb_dij[i] = tuple(cp.asarray(arr) for arr in vkb_dij)
        self.vnl_diag = cp.asarray(self.vnl_diag)
        self.vloc_r = cp.asarray(self.vloc_r)

        self.fft_mod = CpFFT3D(self.gkspc.grid_shape, self.gkspc.idxgrid,
                               normalize_idft=False)

    def h_psi(self, l_psi: cp.ndarray, l_hpsi: cp.ndarray):
        l_hpsi[:] = self.ke_gk * l_psi
        for psi, hpsi in zip(l_psi, l_hpsi):
            psi_r = self.fft_mod.g2r(psi)
            cp.multiply(self.vloc_r[self.idxspin], psi_r, out=psi_r)
            hpsi += self.fft_mod.r2g(psi_r)

        for vkb, dij in self.l_vkb_dij:
            # proj = vkb.conj() @ l_psi.T
            # l_hpsi += (dij @ proj).T @ vkb
            proj = gemm('H', 'N', vkb.T, l_psi.T)
            gemm('N', 'N', vkb.T, dij@proj, l_hpsi.T, 1.0, 1.0)
