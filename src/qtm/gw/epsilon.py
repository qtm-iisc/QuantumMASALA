from functools import lru_cache
from typing import List

import numpy as np

# from qtm import pw_logger
from qtm.constants import RYDBERG
from qtm.crystal import Crystal
from qtm.gspace.fft.full import FFT3DFull
from qtm.gspace.fft.utils import cryst2idxgrid
from qtm.klist import KList
from qtm.gspace.fft import get_fft_driver
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import minimal_grid_shape, GSpace
from qtm.dft.kswfn import KSWavefun
from qtm.gw.core import QPoints
from qtm.gw.io_bgw.epsinp import Epsinp
from qtm.gw.io_bgw.epsmat_read_write import read_mats, write_mats
from qtm.gw.io_bgw.wfn2py import WfnData
from qtm.gw.vcoul import Vcoul
from qtm.gspace.fft.utils import idxgrid2cryst


# #@pw_logger.time('epsilon')
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
        epsinp: Epsinp,
    ):
        """Initialize Epsilon

        Parameters
        ----------
        crystal: Crystal
        gspace: GSpace
        kpts: KList
        kptsq: KList
        l_wfn: List[KSWavefun]
        l_wfnq: List[KSWavefun]
        l_gsp_wfn: List[GkSpace]
        l_gsp_wfnq: List[GkSpace]
        qpts: QPoints
        epsinp: NamedTuple
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

        # Construct list of GSpaceQpts objects, one for each q-point
        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):  # qpts.cryst[:4]:
            self.l_gq.append(
                GkSpace(
                    gwfn=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                    ecutwfn=self.epsinp.epsilon_cutoff * RYDBERG,
                )
            )

        self.vcoul = Vcoul(
            gspace=self.gspace,
            qpts=self.qpts,
            bare_coulomb_cutoff=epsinp.epsilon_cutoff,
            avgcut=0,
        )

        # To store epsilon inverse matrices, for sigma calculation
        self.l_epsinv: dict = {}

    #@pw_logger.time("Epsilon:matrix_elements_rewrite")
    def matrix_elements_rewrite(self, i_q, yielding=False):
        pass

    #@pw_logger.time("Epsilon:matrix_elements")
    def matrix_elements(self, i_q, yielding=False):
        """
        Calculate the plane wave matrix elements required for calculation of polarizability.

            M_{n,n'} (k,q,G) = < n, k+q | exp(i(G+q).r) | n', k >

        Here,
            k is summed over,
            q is given,
            so M_{n_occ,n'_emp}(G)
            where G is ordered and cut-off such that
                |G+q|^2 < epsilon_cutoff
            Also, as per the paper, the states involved in summation
            are such that their eigenvalues are less than Ecut.

            k+q may not lie in the given k-grid so we add displ. vector (umklapp)
            and so we have to subtract that displ. from (q+G) in the central exponential.

        Reference: eqns (8) and (13) in BGW arxiv paper (1111.4429).

        Parameters
        ----------
        i_q : int
            Index of the q-point
        yielding : bool, optional
            If True, yield plane wave matrix elements, one-by-one for each bra (valence) band.
            If False, yield the full plane wave matrix elements at once.

        Yields
        ------
        M: np.ndarray
            Plane wave matrix elements

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
            evl_v = [evl_q[i][:n_v_max] for i in range(len(evl_q))]
        else:
            umklapp = -np.floor(np.around(l_k + qpt, 5))
            l_kplusq = l_k + qpt + umklapp
            evl_v = [evl[i][:n_v_max] for i in range(len(evl))]

        evl_c = [evl[i][:number_bands] for i in range(len(evl))]

        prod_grid_shape = np.prod(self.gspace.grid_shape)

        # For efficiency, we are caching the valence band ifft's to improve serial performance.

        # #@pw_logger.time('matrix_elements:get_evc_gk_r2g')
        @lru_cache(maxsize=int(n_v_max))
        def get_evc_gk_r2g(i_k_v, i_b_v):
            if is_qpt_0:
                wfn_v = self.l_wfnq[i_k_v]
                gkspc = self.l_gsp_wfnq[i_k_v]
            else:
                wfn_v = self.l_wfn[i_k_v]
                gkspc = self.l_gsp_wfn[i_k_v]

            # :gkspc.numgk])
            arr_r = np.zeros(self.gspace.grid_shape, dtype=complex)
            gkspc._fft.g2r(wfn_v.evc._data[i_b_v, :], arr_out=arr_r)
            return arr_r

        # MATRIX ELEMENTS CALCULATION ---------------------------------------------------
        if not yielding:
            M = np.zeros(
                (n_kpts, n_c_max, n_v_max, self.l_gq[i_q].size_g), dtype=complex
            )

        prev_i_k_c = None  # To avoid recalculation for the same kc vector,
        for i_c in range(n_c):  # , desc="mtxel i_c loop"):
            i_k_c = l_i_c[0][i_c]  # unoccupied k indices, repeated
            i_b_c = l_i_c[1][i_c]  # unoccupied band indices, repeated

            # phi_c: Fourier transform of wfn_c to real space
            wfn_c = self.l_wfn[i_k_c]

            # :self.l_gsp_wfn[i_k_c].numgk])
            phi_c = np.zeros(wfn_c.gkspc.grid_shape, dtype=complex)
            # print("wfn_c.evc.shape",wfn_c.evc.shape)
            self.l_gsp_wfn[i_k_c]._fft.g2r(arr_inp=wfn_c.evc._data[i_b_c, :], arr_out=phi_c)

            # To avoid re-calculation for different bands of the same k_c-vector:
            if prev_i_k_c != i_k_c:
                prev_i_k_c = i_k_c

                l_g_umklapp = self.l_gq[i_q].g_cryst - umklapp[i_k_c][:, None]

                idxgrid = cryst2idxgrid(shape=self.gspace.grid_shape, g_cryst=l_g_umklapp.astype(int))

                # check
                umklapped_fft_driver = get_fft_driver()(
                    self.gspace.grid_shape,
                    # grid_g_umklapp,
                    idxgrid,
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

                phi_v = get_evc_gk_r2g(i_k_v, i_b_v)

                prod = np.multiply(np.conj(phi_c), phi_v)
                # Do FFT
                fft_prod = np.zeros(umklapped_fft_driver.idxgrid.shape, dtype=complex)
                umklapped_fft_driver.r2g(prod, fft_prod)
                # The FFT result will be cut-off according to umklapped_fft_driver's cryst
                sqrt_Ec_Ev = np.sqrt(evl_c[i_k_c][i_b_c] - evl_v[i_k_v][i_b_v])

                if yielding:
                    M = fft_prod * np.reciprocal(sqrt_Ec_Ev * prod_grid_shape)
                    yield M
                else:
                    M[i_k_c, i_b_c - n_c_beg, i_b_v] = fft_prod * np.reciprocal(
                        sqrt_Ec_Ev * prod_grid_shape
                    )
                # In `polarizability`, these three indices are being summed over anyway, so it might make sense to have an alternative function for polarizability where polarizability is calculated directly, without ever storing the full M.

        if not yielding:
            yield M

    @classmethod
    def from_data(cls, wfndata: WfnData, wfnqdata: WfnData, epsinp: Epsinp):
        """Construct an `Epsilon` object from objects derived directly from BGW-compatible input files.

        Parameters
        ----------
        wfndata: WfnData
            WfnData object, constructed from WFN.h5 file.
        wfnqdata: WfnData
            WfnData object for shifted grid, constructed from WFNq.h5 file.
        epsinp: Epsinp
            Epsinp object, constructed from epsilon.inp file.
        """

        qpts = QPoints.from_cryst(wfndata.kpts.recilat, epsinp.is_q0, *epsinp.qpts)

        if epsinp.no_min_fftgrid == True or epsinp.epsilon_cutoff is None:
            gspace = wfndata.grho
            l_gk = wfndata.l_gk
            l_gk_q = wfnqdata.l_gk
        else:
            eps_fftgrid_ecut = wfndata.l_wfn[0].gkspc.ecutwfn + epsinp.epsilon_cutoff
            epsilon_grid_shape = minimal_grid_shape(
                wfndata.crystal.recilat, eps_fftgrid_ecut
            )
            gspace = GSpace(wfndata.crystal.recilat, eps_fftgrid_ecut, epsilon_grid_shape)
            l_gk = [
                GkSpace(gspace, k_cryst, wfndata.l_wfn[0].gkspc.ecutwfn)
                for k_cryst in wfndata.kpts.cryst
            ]
            l_gk_q = [
                GkSpace(gspace, k_cryst, wfndata.l_wfn[0].gkspc.ecutwfn)
                for k_cryst in wfnqdata.kpts.cryst
            ]

        return Epsilon(
            wfndata.crystal,
            gspace,
            wfndata.kpts,
            wfnqdata.kpts,
            wfndata.l_wfn,
            wfnqdata.l_wfn,
            l_gk,
            l_gk_q,
            qpts,
            epsinp,
        )

    # CHI ===========================================================================

    #@pw_logger.time("Epsilon:polarizability")
    def polarizability(self, M):
        """Polarizability Matrix

        Parameters
        ----------
        M : np.ndarray
            Plane wave matrix elements.
        """

        # Nice suggestion by SH: consider doing this multiplication within matrix_elements, to avoid using excess space.
        polarizability_matrix = -np.einsum("ijkl,ijkm->lm", np.conj(M), M)

        return (
            (4*13824**4) * polarizability_matrix / (self.crystal.reallat.cellvol * self.kpts.numk)
        )

    #@pw_logger.time("Epsilon:polarizability_active")
    def polarizability_active(self, i_q):
        """Calculates Polarizability Matrix.
        Differs from the ``polarizability`` method in that it actively calls ``matrix_element`` function and does the sums in a memory-efficient way.

        Parameters
        ----------
        i_q : int
            index of qpoint for which polarizability will be calculated.
        """

        polarizability_matrix = np.zeros(
            (self.l_gq[i_q].numgk, self.l_gq[i_q].numgk), dtype=complex
        )
        for M in self.matrix_elements(i_q=i_q, yielding=True):
            polarizability_matrix += -np.einsum("l,m->lm", np.conj(M), M)

        return (
            (4*13824**4) * polarizability_matrix / (self.crystal.reallat.cellvol * self.kpts.numk)
        )

    # EPSILON INVERSE ===========================================================================
    #@pw_logger.time("Epsilon:epsilon_inverse")
    def epsilon_inverse(self, i_q, polarizability_matrix, store=True):
        """Calculate epsilon inverse, given index of q-point

        Parameters
        ----------
        i_q : int
            index of q-point
        polarizability_matrix : np.ndarray
        """

        vqg = self.vcoul.v_bare(i_q)
        I = np.identity(len(polarizability_matrix))

        eps = I - np.einsum("j,ij->ij", vqg, polarizability_matrix, optimize=True)

        epsinv = np.linalg.inv(eps)

        if store:
            self.l_epsinv[i_q] = epsinv

        return epsinv

    # READ / WRITE EPSMAT =============================================================

    #@pw_logger.time("Epsilon:write_epsmat")
    def write_epsmat(self, filename: str, epsinvmats: List[np.ndarray]):
        """Save data to epsmat.h5 file

        Parameters
        ----------
        filename : str
        epsinvmats : List[np.ndarray]
            List of epsilon inverse matrices to be saved.
        """
        if len(epsinvmats) > 0:
            write_mats(filename, epsinvmats)

    def read_epsmat(self, filename: str):
        """Read epsmat from epsmat.h5 file

        Parameters
        ----------
        filename : str

        Returns
        -------
        epsinvmats : List[np.ndarray]
            List of epsilon inverse matrices read from file.
        """
        return read_mats(filename)
