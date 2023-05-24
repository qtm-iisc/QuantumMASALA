from functools import lru_cache
from pprint import pprint
from time import time, time_ns
from typing import List, NamedTuple
import numpy as np
from tqdm import trange
from quantum_masala.constants import RYDBERG
from quantum_masala.core import (
    Crystal,
    GSpace,
    KList,
)
from quantum_masala.core.fft import get_fft_driver
# import gc

# from quantum_masala.core.fft import FFTGSpace
from quantum_masala.core.gspc.gkspc import GkSpace
from quantum_masala.dft.kswfn import KSWavefun

# import io_bgw.inp
# from core import QPoints, GSpaceQpt
from quantum_masala.gw.core import QPoints, sort_cryst_like_BGW
from quantum_masala.gw.io_bgw.epsmat_read_write import read_mats, write_mats
from quantum_masala.gw.vcoul import Vcoul


class Epsilon:
    """Epsilon Matrix Class
    Generate (optionally frequency dependent) dielectric function and its inverse.

    Attributes
    ----------
    epsmat: np.ndarray
        Dielectric function matrix
    epsinv: np.ndarray
        Inverse of the Dielectric function matrix
    chi0: np.ndarray
        Irreducible Polarizability matrix

    Methods
    -------
    __init__(self,...) -> None
        Does the core calculation to construct epsmat, epsinv, chi0
    write_epsmat(self, filename) -> None
        Write the Dielectric function matrix to an HDF5 file

    - Calculate Planewave Matrix elements : Bottleneck FFT O(N3)
    - Calculate RPA polarizability matrix : Bottleneck Matrix Multiplication O(N4)
    - Calculate Epsilon and Epsilon Inverse matrix: O(N3)

    """

    TOLERANCE = 1e-5

    # TODO: Init should (optionally) accept Vcoul etc. and not have to construct it. (because minibz averaging is expensive.)

    def __init__(
        self,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        kptsq: KList,
        l_wfn: List[KSWavefun],
        l_wfnq: List[KSWavefun],
        l_gsp_wfn: List[GkSpace],
        l_gsp_wfnq: List[GkSpace],
        qpts: QPoints,
        epsinp: NamedTuple,
    ):
        """Initialize Epsilon

        - Receive GSpace, ElectronWfn etc. objects constructed from ``wfn.h5`` and ``wfnq.h5``
        - Load EpsilonInp object constructed from ``epsilon.inp``

        """
        self.crystal = crystal
        self.gspace = gspace
        self.kpts = kpts
        self.kptsq = kptsq
        self.l_wfn = l_wfn
        self.l_wfnq = l_wfnq
        self.l_gsp_wfn = l_gsp_wfn
        self.l_gsp_wfnq = l_gsp_wfnq
        self.qpts = qpts
        self.epsinp = epsinp

        # Construct list of GSpaceQpts objects, one for each qpt
        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):  # qpts.cryst[:4]:
            self.l_gq.append(
                GkSpace(
                    gspc=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                    ecutwfc=self.epsinp.epsilon_cutoff
                    * RYDBERG, 
                )
            )

        self.vcoul = Vcoul(
            gspace=self.gspace,
            qpts=self.qpts,
            bare_coulomb_cutoff=epsinp.epsilon_cutoff,
            avgcut=0,
        )

    def matrix_elements(self, i_q, yielding=False):
        """
        To Calculate the M - matrix for calculation of polarizability.

            M_{n,n'} (k,q,G) = < n, k+q | exp(i(G+q).r) | n', k >

        Here,
            k is summed over,
            q is given,
            so M_{n_occ,n'_emp}(G)
            where G is ordered and cut-off such that
                |G+q|^2 < epsilon_cutoff
            Also, as per the paper, the states involved in summation
            are such that their eigenvalues are less than Ecut.

            NOTE: k+q may not lie in the given k-space so we add displ. vector (umklapp)
            and so we have to subtract that displ. from (q+G) in the central exponential.

        Refer: eqns (8) and (13) in BGW arxiv paper (1111.4429).
        """

        # ** ALL K VECTORS, Q VECTORS, G VECTORS WILL BE IN CRYSTAL BASIS **

        # k points data

        n_kpts = self.kpts.numk  # kpoints.nrk
        evl = []  # kpoints.el
        evl_q = []
        occ_all_bands = []  # kpoints.occ

        for i_k in range(n_kpts):
            occ_all_bands.append(self.l_wfn[i_k].occ[0])
            evl.append(self.l_wfn[i_k].evl)
            evl_q.append(self.l_wfnq[i_k].evl)

        # List of k-points in crystal coords
        l_k = self.kpts.cryst  # kpoints.rk

        # Wavefunction data
        gvecs = [self.l_wfn[i_k].gkspc.g_cryst for i_k in range(n_kpts)]

        # Load epsilon.inp data
        number_bands = self.epsinp.number_bands
        is_q0 = self.qpts.is_q0  # np.array(inp.qpts[:, 4], dtype=bool)
        l_q = self.qpts.cryst  # list of q-points in crystal coords

        # Occupation numbers
        occ_all_bands = np.array(occ_all_bands)

        occ = occ_all_bands[:, 0:number_bands]
        # ^ indices for reference: [index of kpoint, band index]
        l_i_v = np.where(occ == 1)  # list of indices of occupied   bands
        l_i_c = np.where(occ == 0)  # list of indices of unoccupied bands

        n_c = len(l_i_c[0])  # Number of Unoccupied (Conduction) bands

        n_v_max = max(l_i_v[1]) + 1
        n_c_beg = min(l_i_c[1])
        n_c_max = number_bands - min(l_i_c[1])

        # Map G vectors to points in size_1 x size_2 x size_3 cube for all k points
        # returns nrk x 3 x ng_k (note that ng_k is not same for all k)
        # Could be made simpler by extending to ngkmax and making np.array
        l_i_gmapped = []
        for k in range(n_kpts):
            id_k = np.transpose(
                np.mod(gvecs[k], np.array(self.gspace.grid_shape)[:, None]).astype(int)
            )
            l_i_gmapped.append(id_k)

        qpt = l_q[i_q]
        is_qpt_0 = is_q0[i_q]

        # UMKLAPP ----------------------------
        # Cycle to peridic image if k-q lies outside k grid (in crystal coords).
        # umklapp is the Correction that is to be added to bring the vector to 1st +BZ

        if is_qpt_0:
            umklapp = -np.floor(np.around(l_k, 5))
            l_kplusq = l_k + umklapp
            evl_v = [evl_q[_][:n_v_max][0] for _ in range(len(evl_q))]
        else:
            umklapp = -np.floor(np.around(l_k + qpt, 5))
            l_kplusq = l_k + qpt + umklapp
            evl_v = [evl[_][:n_v_max][0] for _ in range(len(evl))]

        evl_c = [evl[_][:number_bands][0] for _ in range(len(evl))]

        prod_grid_shape = np.prod(self.gspace.grid_shape)

        # For efficiency, caching the valence band ifft's to improve serial performance,
        # but this should be removed later, or a cleaner solution must be found.
        @lru_cache(maxsize=int(n_v_max))
        def get_evc_gk_r2g(i_k_v, i_b_v):
            if is_qpt_0:
                wfn_v = self.l_wfnq[i_k_v]
            else:
                wfn_v = self.l_wfn[i_k_v]
            return wfn_v.gkspc.fft_mod.g2r(wfn_v.evc_gk[0, i_b_v, :])

        # MATRIX ELEMENTS CALCULATION ---------------------------------------------------
        if not yielding:
            M = np.zeros((n_kpts, n_c_max, n_v_max, self.l_gq[i_q].numgk), dtype=complex)
        # Memory reqd = complex(8bytes) * n_kpts * n_c_max * n_v_max * self.l_gq[i_q].numgk
        # For a converged run, such as the one shown in the QTM paper,
        # n_kpts ~ 3
        # n_c_max ~ 272-4
        # n_v_max ~ 4
        # numgk ~ 537

        # TODO: Refactor older i_c, i_v loops to just one i_k_c loop,
        #       and (depending on target architecture,) vectorize i_b_c and i_b_v to compute all ffts at once and reuse them.
        #       This way, the r2g is expected to be the most frequently called routine.
        #       r2g will be called nv*nc times, whereas g2r will be called nv+nc times
        #       The details will depend on the nature of parallelization.
        #       And the analysis will be slightly different in the case of Sigma's matrix elements.

        prev_i_k_c = None  # To avoid recalculation for the same kc vector,
        for i_c in range(n_c):#, desc="mtxel i_c loop"):
            i_k_c = l_i_c[0][i_c]  # unoccupied k indices, repeated
            i_b_c = l_i_c[1][i_c]  # unoccupied band indices, repeated

            # phi_c: Fourier transform of wfn_c to real space
            wfn_c = self.l_wfn[i_k_c]
            phi_c = wfn_c.gkspc.fft_mod.g2r(wfn_c.evc_gk[0, i_b_c, :])

            # To avoid re-calculation for different bands of the same k_c-vector:
            if prev_i_k_c != i_k_c:
                prev_i_k_c = i_k_c

                l_g_umklapp = self.l_gq[i_q].g_cryst - umklapp[i_k_c][:, None]

                grid_g_umklapp = tuple(
                    np.mod(
                        l_g_umklapp, np.array(self.gspace.grid_shape)[:, None]
                    ).astype(int)
                )

                umklapped_fft_driver = get_fft_driver()(
                    self.gspace.grid_shape,
                    grid_g_umklapp,
                    normalise_idft=False,  
                    # NOTE: normalise_idft=False will be the default for all gw code,
                    # as this is the default for gkspc.fft_driver constructor call.
                    # However, it matters only for ifft, i.e. g2r,
                    # So not relevant for umklapped_fft_driver.
                )

            # Obtain a list of indices of valence kpoints that match  k_c + q
            l_k_v = l_k[l_i_v[0][:], :]  # [ k-index , component index: 0,1,2 ]
            kcplusq = l_kplusq[i_k_c]
            l_i_match = np.nonzero(
                np.all(
                    np.isclose(l_k_v, kcplusq[None, :], atol=Epsilon.TOLERANCE), axis=1
                )
            )[0]

            # for k_v == k_c + qvec:
            for i_v in l_i_match:
                i_k_v = l_i_v[0][i_v]  # occupied k indices, repeated
                i_b_v = l_i_v[1][i_v]  # occupied band indices

                # if is_qpt_0:
                #     wfn_v = self.l_wfnq[i_k_v]
                # else:
                #     wfn_v = self.l_wfn[i_k_v]

                # phi_v = wfn_v.gkspc.fft_mod.g2r(wfn_v.evc_gk[0, i_b_v, :])
                phi_v = get_evc_gk_r2g(i_k_v, i_b_v)

                prod = np.multiply(np.conj(phi_c), phi_v)

                # Do FFT
                fft_prod = umklapped_fft_driver.r2g(prod)

                # The FFT result will be cut-off according to umklapped_fft_driver's cryst
                sqrt_Ec_Ev = np.sqrt(evl_c[i_k_c][i_b_c] - evl_v[i_k_v][i_b_v])

                if yielding:
                    M = fft_prod * np.reciprocal(sqrt_Ec_Ev * prod_grid_shape)
                    yield M
                else:
                    M[i_k_c, i_b_c - n_c_beg, i_b_v] = fft_prod * np.reciprocal(
                        sqrt_Ec_Ev * prod_grid_shape
                    )
                # In `polarizability`, these three indices are bein summed over anyway, so it might make sense to have an alternative function for polarizability where polarizability is calculated directly, without ever storing the full M.
                
        if not yielding:
            yield M

    @classmethod
    def from_data(cls, wfndata, wfnqdata, epsinp):
        qpts = QPoints.from_cryst(wfndata.kpts.recilat, epsinp.is_q0, *epsinp.qpts)

        return Epsilon(
            wfndata.crystal,
            wfndata.grho,
            wfndata.kpts,
            wfnqdata.kpts,
            wfndata.l_wfn,
            wfnqdata.l_wfn,
            wfndata.l_gk,
            wfnqdata.l_gk,
            qpts,
            epsinp,
        )

    # CHI ===========================================================================

    def polarizability(self, M):
        """Polarizability Matrix in G, G' both ordered by |G+q|"""

        # Nice suggestion by SH: consider doin this within matrix_elements, to avoid space wastage.
        polarizability_matrix = -np.einsum("ijkl,ijkm->lm", np.conj(M), M)

        return polarizability_matrix / (self.crystal.reallat.cellvol * self.kpts.numk)

    def polarizability_active(self, i_q):
        """Returns Polarizability Matrix in G, G' both ordered by |G+q|. Differs from the ``polarizability`` method in that it actively calls ``matrix_element`` function and does the sums in a memory-efficient way."""
        # for i_q in trange(0, self.qpts.numq, desc="Epsilon> q-pt index"):
        polarizability_matrix = np.zeros((self.l_gq[i_q].numgk, self.l_gq[i_q].numgk), dtype=complex)
        for M in self.matrix_elements(i_q=i_q, yielding=True):
            polarizability_matrix += -np.einsum("l,m->lm", np.conj(M), M)
            
        return polarizability_matrix / (self.crystal.reallat.cellvol * self.kpts.numk)

    # EPSILON INVERSE ===========================================================================
    def epsilon_inverse(self, i_q, polarizability_matrix):
        """Calculate epsilon inverse, given index of q-point"""

        vqg = self.vcoul.v_bare(i_q)
        I = np.identity(len(polarizability_matrix))

        eps = I - np.einsum("j,ij->ij", vqg, polarizability_matrix, optimize=True)

        return np.linalg.inv(eps)

    # READ / WRITE EPSMAT =============================================================

    def write_epsmat(self, filename, epsinvmats):
        """Save data to epsmat.h5 file"""
        write_mats(filename, epsinvmats)

    def read_epsmat(self, filename):
        """Read epsmat from epsmat.h5 file"""
        return read_mats(filename)
