"""
Sigma
=====
File used for the purpose of generating gpp data for the paper.

FIXME WARNING: for Matrix elements, the number_bands from sigma should be used for n and n'
but then how will n" number of bands be found? IDK perhaps number_bands and band_min and band_max are different when this happens.
But for speed, here we have used 8 bands and used number_bands for n".
Do FIX this later.
"""


from quantum_masala.gw.mydebugtoolkit import read_txt
from quantum_masala.gw.vcoul import Vcoul
from quantum_masala.core.fft import get_fft_driver
from quantum_masala.dft.kswfn import KSWavefun
from quantum_masala.core import AtomBasis, Crystal, GSpace, KList, RealLattice
from quantum_masala.gw.core import QPoints  # GSpaceQpt,
from typing import List, NamedTuple
from copy import deepcopy
import time
import sys
from quantum_masala.gw.h5_io.h5_utils import *
import numpy as np
from functools import lru_cache
from tqdm import trange, tqdm
import datetime
from pprint import pprint
import warnings
from quantum_masala.constants import ELECTRONVOLT_RYD, RYDBERG_HART
from quantum_masala.core.gspc.gkspc import GkSpace
from quantum_masala.gw.core import sort_cryst_like_BGW

print(
    "Sigma script started running : ",
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
)


# Imports

# from mydebugtoolkit import *
sys.path.append("..")
sys.path.append(".")
# from ..src.fft.fft_driver import FFT


# -------- Constant Parameters -------------------
NDIGITS = 5
TOLERANCE = 1e-5

# -------- Debugging params ----------------------
global_test_vcoul = False  # True may not work for now, due to broken sorted-gpts
global_test_epsinv = global_test_vcoul


# dirname = "./test/bgw/"
dirname = "./scripts/results/si_4_gw_cohsex_nn25000/"  # For GPP: _gpp_skipw0/"
# old_dirname = "../QE_data/control_scripts/"
old_dirname = dirname

# Sigma Class


# from time import time, time_ns


# import io_bgw.inp

# from quantum_masala.core.fft import FFTGSpace


class Sigma:
    """Sigma Matrix Class
    Generates self-energy matrix.

    Attributes
    ----------

    Methods
    -------

    """

    ryd = 13.605692530000001
    limitone = 1 / TOLERANCE / 4  # limitone=1D0/(TOL_Small*4D0)
    limittwo = 0.25  # limittwo=sig%gamma**2 what is gamma?
    fixwings = not global_test_epsinv

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
        # sigma_kpts: KList,
        sigmainp: NamedTuple,
        epsinp: NamedTuple,
        l_epsmats: List[np.ndarray],
    ):
        """Initialize Sigma

        - Receive GSpace, ElectronWfn etc. objects constructed from ``wfn.h5`` and ``wfnq.h5``
        - Load SigmaInp object constructed from ``sigma.inp``

        Parameters
        ----------

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
        self.sigmainp = sigmainp
        self.epsinp = epsinp
        self.l_epsmats = l_epsmats

        # Read RHO.dat file and create hash map for rhogrid
        # TODO: Have the ability to optionally generate `rho` directly from wfndata
        self.rho = inp.read_rho(old_dirname + "RHO")
        rhogvecs = self.rho.gvecs
        rhogrid = self.gspace.grid_shape
        self.rho_gvecs_hashed = (
            (rhogvecs[:, 0] + rhogrid[0] // 2)
            + (rhogvecs[:, 1] + rhogrid[1] // 2) * rhogrid[0]
            + (rhogvecs[:, 2] + rhogrid[2] // 2) * rhogrid[0] * rhogrid[1]
        )
        self.rho_dict = dict(
            zip(self.rho_gvecs_hashed, list(range(len(self.rho_gvecs_hashed))))
        )

        # Constants
        Sigma.sigma_factor = Sigma.ryd / (crystal.reallat.cellvol * qpts.numq)
        print("Sigma.sigma_factor", Sigma.sigma_factor)

        # K-Points (typically from SigmaInp)
        self.l_k_indices = []
        for kpt in self.sigmainp.kpts[:, :3]:
            self.l_k_indices.append(
                np.where(np.all(kpt == self.kpts.cryst, axis=1))[0][0]
            )

        self.l_k_indices = np.array(self.l_k_indices)
        self.n_kpts = len(self.l_k_indices)
        self.slice_l_k = np.s_[self.l_k_indices]
        print("self.l_k_indices", self.l_k_indices)

        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):  # qpts.cryst[:4]:
            self.l_gq.append(
                GkSpace(
                    gspc=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                    ecutwfc=self.epsinp.epsilon_cutoff * RYDBERG_HART,
                )
            )

        self.sorted_gpts = None

        # self.l_gq_argsort_gqnorm2 = [
        #     self.l_gq[i_q].gk_indices_tosorted for i_q in range(self.qpts.numk)
        # ]

        # self.l_gq_inverse_argsort_gqnorm2 = [
        #     self.l_gq[i_q].gk_indices_fromsorted for i_q in range(self.qpts.numk)
        # ]

        # FIXME: Sorting needs to be turned off in vcoul as well.
        self.vcoul = Vcoul(
            gspace=self.gspace,
            qpts=self.qpts,
            bare_coulomb_cutoff=sigmainp.bare_coulomb_cutoff,
        )
        print("vcoul:", self.vcoul)
        self.vcoul.calculate_vcoul(
            averaging_func=self.vcoul.v_minibz_montecarlo_hybrid
        )  # v_minibz_sphere)

        # "Epsilon Inverse minus Identity" matrices
        self.l_epsinv_I = []

        # First convert epsmats G-vec ordering
        # which is assumed to be that of BGW
        # i.e., even when we exported epsmats, it was our (qtm's) duty to make G-vec ordered as per BGW
        # for i_q in range(self.qpts.numq):
        #     # print("i_q", i_q)
        #     self.l_epsmats[i_q] = reorder_2d_matrix_sorted_gvecs(self.l_epsmats[i_q], self.l_gq[i_q].gk_indices_fromsorted)

        for i_q in range(self.qpts.numq):
            # print("i_q", i_q)
            epsinv = self.l_epsmats[i_q]  # [0,0]

            # Fix wings (look at Table 2 in BerkeleyGW Paper (2012))
            if self.fixwings:
                # print("before fixwings")
                # print(epsinv[1])
                # print(epsinv[self.l_gq[i_q].gk_indices_tosorted][1][self.l_gq[i_q].gk_indices_tosorted])
                epsinv = self.vcoul.calculate_fixedeps(
                    epsinv, i_q, random_sample=False)
                # print(epsinv[1])
                # print("after fixwings")

            epsinv_I = epsinv - np.eye(len(epsinv))
            self.l_epsinv_I.append(
                reorder_2d_matrix_sorted_gvecs(
                    epsinv_I, self.l_gq[i_q].gk_indices_fromsorted
                )
            )

        # Deprecated code, used earlier for manual parallelization
        # of sigma calculation over q-points
        l_q = deepcopy(self.qpts.cryst)
        if len(sys.argv) > 3:
            n_segments = int(sys.argv[2])
            q_n = int(sys.argv[3])
            self.q_loop_beg = (len(l_q) // n_segments) * q_n
            self.q_loop_end = (len(l_q) // n_segments) * (q_n + 1)
        else:
            self.q_loop_beg = 0
            self.q_loop_end = len(l_q)

        return

    @classmethod
    def from_data(cls, wfndata, wfnqdata, sigmainp, epsinp, l_epsmats):
        # FIXME: Needs more robust logic, i.e. read qpts from epsmat.h5 etc. as done by BGW/Sigma
        #        That is, we shouldn't require epsilon.inp file for Sigma calculation.
        #        Ideally, epsmat.h5 should contain all the data that we need,
        #        thus ensuring consistency of data.
        l_qpts = np.array(epsinp.qpts)
        # print("l_qpts",l_qpts)
        l_qpts[0] *= 0
        qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *l_qpts)
        # qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *sigmainp.kpts)

        return Sigma(
            wfndata.crystal,
            wfndata.grho,
            wfndata.kpts,
            wfnqdata.kpts,
            wfndata.l_wfn,
            wfnqdata.l_wfn,
            wfndata.l_gk,
            wfnqdata.l_gk,
            qpts,
            sigmainp,
            epsinp,
            l_epsmats,
        )

    # ==================================================================
    # Helper methods

    def rhohash(self, gvec):
        """Hashing function for G-grid vectors.

        Parameters
        ----------
        gvec
            G-space vector, in crystal coordinates, of shape (3,).

        Returns
        -------
            A unique interger for G-vectors on the G-grid
        """
        rhogrid = self.gspace.grid_shape
        return (
            (gvec[0] + rhogrid[0] // 2)
            + (gvec[1] + rhogrid[1] // 2) * rhogrid[0]
            + (gvec[2] + rhogrid[2] // 2) * rhogrid[0] * rhogrid[1]
        )

    def index_minusq(self, i_q):
        """Given index of some 'qpt', return index corresponding to '1 - qpt'.

        Parameters
        ----------
        i_q
            index of qvec from self.qpts

        Returns
        -------
            index of 1-qvec in the same list, i.e. in self.qpts.cryst
        """

        l_q = self.qpts.cryst  # list of q-points in crystal coords
        qpt = l_q[i_q]

        # Find index of -q
        if np.linalg.norm(qpt) < TOLERANCE:  # or self.qpts.index_q0 == i_q:
            # print("self.qpts.index_q0 == i_q")
            i_minusq = i_q
        else:
            ZERO_TOL = 1e-5
            target = np.zeros_like(qpt)
            for d in range(3):
                if abs(qpt[d]) > ZERO_TOL:
                    target[d] = 1 - qpt[d]  # Core task of this routine

            sub_norm = np.linalg.norm(l_q - target[None, :], axis=1)
            i_minusq = np.argmin(sub_norm, axis=0)

        return i_minusq

    def map_g_to_g_minusq(self, g, g_minusq, q, minusq):
        """Return the indices in g_minusq that corresponds to respective -g vectors
        Consider making a hash-map to reduce search time.

        Parameters
        ----------
        g
            List of G-vectors
        g_minusq

        q

        minusq


        Returns
        -------
            Indices in g_minusq that corresponds to respective -g vectors.
        """
        # print("map_g_to_g_minusq: started", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        start_time = time.time_ns()

        # Initialize indices to -1's, which will indicate "not found" case
        # Not found will never occur because we are using np.argmin to get an index
        # Still, we need to initialize with something,
        # and it's not a bad idea to initialize with -1
        indices = -1 * np.ones(len(g), dtype=int)

        for i, gvec in enumerate(g):
            target = -1 * (gvec + q)
            sub_norm = np.linalg.norm(
                (g_minusq + minusq) - target[None, :], axis=1)
            i_g_minusq = np.argmin(sub_norm, axis=0)
            if sub_norm[i_g_minusq] > 1e-5:
                print("Problem in map_g_to_g_minusq", q, sub_norm[i_g_minusq])
                print("g_minusq[i_g_minusq]", g_minusq[i_g_minusq])
                print("minusq", minusq)
            indices[i] = i_g_minusq

        # print("map_g_to_g_minusq: ended  ", (time.time_ns() - start_time) / 1e6)
        return indices

    def find_indices(self, l_targets, l_list):
        """Find indices of elements matching the targets in list.
        Returns list of indices corresponding to target vector. -1 if not found"""
        # print("find_indices: started", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # start_time = time.time_ns()
        ret = []
        for target in l_targets:
            if target in l_list:
                ret.append(l_list.index(target))
            else:
                ret.append(-1)
        # print("find_indices: ended  ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # print("find_indices: ended  ", (time.time_ns()-start_time)/1e6)
        return ret

    # ==================================================================
    # Plane wave matrix calculation methods

    def matrix_elements(
        self,
        i_q,
        bra_all_bands=False,
        ket_all_bands=False,
        ret_E=False,
        yielding=False,
        l_k_indices=None,
    ):
        """
        Parameters
        ----------
        i_q : index of q-point
        bra_all_bands : If True, calculate mtxel for all bands for bra. Otherwise, only occupied bands
        ket_all_bands : If True, calculate mtxel for all bands for ket. Otherwise, only unoccupied bands
        ret_E : If True, return energy eigenvalues. Thisoption is useful for GPP Sigma, where energy eigenvalues corresponding to matrix elements are required.

        To Calculate the M - matrix for calculation of polarizability.

            M_{n,n'} (k,q,G) = < n, k+q | exp(i(G+q).r) | n', k >

        Here,
            k will be summed over in Chi,
            q is given,
            so M_{n_occ,n'_emp}(G)
            where G is ordered and cut-off such that
                |G+q|^2 < epsilon_cutoff
            so the set {G} depends on q and will be the same for each mtxel call.
            Also, as per the paper, the states involved in summation
            are such that their eigenvalues are less than Ecut.

            NOTE: k+q may not lie in the given k-space so we add displ. vector (umklapp)
            and so we have to subtract that displ. from (q+G) in the central exponential.

        Refer: eqns (8) and (13) in BGW arxiv paper (1111.4429).


        """

        # k points data -------------------------------------------------

        if l_k_indices is None:
            l_k_indices = self.l_k_indices

        # list of k-points in crystal coords
        l_k = self.kpts.cryst  # kpoints.rk
        # print("l_k:",l_k)

        # Load Sigma.inp data
        # number_bands = self.sigmainp.number_bands
        is_q0 = self.qpts.is_q0  # np.array(inp.qpts[:, 4], dtype=bool)
        l_q = self.qpts.cryst  # list of q-points in crystal coords

        # Occupation numbers
        
        occ_all_bands = []
        for i_k in range(self.kpts.numk):  # self.l_k_indices:
            occ_all_bands.append(self.l_wfn[i_k].occ[0])
        occ_all_bands = np.array(occ_all_bands)
        occ = occ_all_bands[:, 0:self.sigmainp.number_bands]
        # ^ indices for reference: [index of kpoint, band index]
        l_i_v = np.where(occ == 1)  # list of indices of occupied   bands
        l_i_c = np.where(occ == 0)  # list of indices of unoccupied bands

        prod_grid_shape = np.prod(self.gspace.grid_shape)

        n_v_max = max(l_i_v[1]) + 1
        i_c_beg = min(l_i_c[1])
        n_c_max = self.sigmainp.number_bands - min(l_i_c[1])

        qpt = l_q[i_q]
        is_qpt_0 = None if is_q0 == None else is_q0[i_q]

        if is_qpt_0:
            umklapp = -np.floor(np.around(l_k, 5))
            l_kplusq = l_k + umklapp
        else:
            umklapp = -np.floor(np.around(l_k + qpt, 5))
            l_kplusq = l_k + qpt + umklapp

        # For efficiency, caching the valence band ifft's to improve serial performance,
        # but this should be removed later, or a cleaner solution must be found.
        @lru_cache()#maxsize=int(number_bands*len(l_k_indices)*2))
        def get_evc_gk_r2g(i_k, i_b, checkq0 = True):
            if is_qpt_0 and checkq0:
                wfn = self.l_wfnq[i_k]
            else:
                wfn = self.l_wfn[i_k]
            return wfn.gkspc.fft_mod.g2r(wfn.evc_gk[0, i_b, :])

        # MATRIX ELEMENTS CALCULATION -------------------

        # Handling `_all_bands` parameters
        if bra_all_bands:
            n_bra =  self.sigmainp.number_bands
            i_b_bra_beg = 0  # Index where valence bands begin (always 0)
        else:
            n_bra = n_v_max
            i_b_bra_beg = 0

        if ket_all_bands:
            # i_b_ket_beg =  self.sigmainp.band_index_min - 1  # Index where conduction bands begin
            # n_ket = self.sigmainp.band_index_max - self.sigmainp.band_index_min + 1
            i_b_ket_beg = 0
            n_ket = self.sigmainp.number_bands
        else:
            i_b_ket_beg = i_c_beg
            n_ket = n_c_max #self.sigmainp.band_index_max - (i_b_ket_beg+1) +1

        # Init M
        if yielding:
            M = np.zeros(
                (self.n_kpts, n_ket, self.l_gq[i_q].numgk), dtype=complex)
        else:
            M = np.zeros(
                (self.n_kpts, n_bra, n_ket,
                 self.l_gq[i_q].numgk), dtype=complex
            )

        # Find pairs of ket and bra k-points
        # They are related as follows:
        # k_ket + qpt = k_bra
        pairs_i_k = []
        for i_k_ket in l_k_indices:
            k_bra_indices = np.where((l_k == l_kplusq[i_k_ket]).all(axis=1))[0]
            assert (
                len(k_bra_indices) > 0
            ), "Could not find k-point in wavefunction k-points that matches k+q"
            pairs_i_k.append((i_k_ket, k_bra_indices[0]))
        # print("pairs_i_k", pairs_i_k)

        if ret_E == True:
            E_ket = np.zeros((self.n_kpts, n_ket), dtype=complex)
            E_bra = np.zeros((self.n_kpts, n_bra), dtype=complex)
            for i_k_ket, i_k_bra in pairs_i_k:
                i_k_ket_in_mtxel_call = np.where(
                    self.l_k_indices == i_k_ket)[0][0]
                for i_b_bra in range(n_bra):
                    E_bra[i_k_ket_in_mtxel_call, i_b_bra] = self.l_wfn[i_k_bra].evl[0][i_b_bra + i_b_bra_beg]
                    
                for i_b_ket in range(n_ket):
                    E_ket[i_k_ket_in_mtxel_call, i_b_ket] = self.l_wfn[i_k_ket].evl[0][i_b_ket + i_b_ket_beg]

        # Matrix elements calculation

        phi_bra = {}
        phi_ket = {}

        for i_b_bra in range(n_bra):
            for i_k_ket, i_k_bra in pairs_i_k:
                i_k_ket_in_mtxel_call = np.where(l_k_indices == i_k_ket)[0][0]
                # wfn_ket = self.l_wfn[i_k_ket]

                l_g_umklapp = self.l_gq[i_q].g_cryst - umklapp[i_k_ket][:, None]

                grid_g_umklapp = tuple(
                    np.mod(l_g_umklapp, np.array(self.gspace.grid_shape)[:, None]).astype(int)
                )

                umklapped_fft_driver = get_fft_driver()(
                    self.gspace.grid_shape,
                    grid_g_umklapp,
                    normalise_idft=False,
                )
                # NOTE: normalise_idft=False will be the default for all gw code,
                # as this is the default for gkspc.fft_driver constructor call.
                # However, it matters only for ifft, i.e. g2r,
                # So not relevant for umklapped_fft_driver.

                phi_bra[i_b_bra] = get_evc_gk_r2g(
                    i_k_bra, i_b_bra + i_b_bra_beg, checkq0=True) / prod_grid_shape

                
                for i_b_ket in range(n_ket):
                    
                    phi_ket[i_b_ket] = get_evc_gk_r2g(
                        i_k_ket, i_b_ket + i_b_ket_beg, checkq0=False) / prod_grid_shape

                    prod = np.multiply(
                        np.conj(phi_ket[i_b_ket]), phi_bra[i_b_bra])
                    fft_prod = umklapped_fft_driver.r2g(prod)
                    if yielding:
                        M[i_k_ket_in_mtxel_call, i_b_ket] = prod_grid_shape * fft_prod
                    else:
                        M[i_k_ket_in_mtxel_call, i_b_bra,
                            i_b_ket] = prod_grid_shape * fft_prod
            if yielding:
                if ret_E == True:
                    yield M, E_bra[:,i_b_bra], E_ket
                else:
                    yield M
            
        if not yielding:
            if ret_E == True:
                yield M, E_bra, E_ket
            else:
                yield M

    def matrix_elements_sigma_exact(self, l_g, ket_all_bands=True, bra_all_bands=True):
        """
        To Calculate the matrix elements for exact sigma_CH.

            M_{n,n'} (k,q,G) = < v, k | exp(i(G'-G).r) | c, k >

        Returns M with shape (n_valence, n_conduction, number of G-vectors within cutoff for q=0)

        """
        # print("Matrix elements Sigma CH Exact")

        # k points data -------------------------------------------------
        # ** ALL K POINTS, Q POINTS, G POINTS IN CRYSTAL BASIS **
        n_kpts = self.qpts.numq
        evl = []
        evl_q0 = []
        occ_in = []

        for i_k in range(n_kpts):
            occ_in.append(self.l_wfn[i_k].occ[0])
            evl.append(self.l_wfn[i_k].evl)
            evl_q0.append(self.l_wfnq[i_k].evl)

        # list of k-points in crystal coords
        l_k = self.kpts.cryst  # kpoints.rk

        # wavefunction data ----------------------------------------------
        # gvecs = [self.l_wfn[i_k].gwfc.cryst for i_k in range(n_kpts)]

        # Load Sigma.inp data --------------------------------------------
        number_bands = sigmainp.number_bands
        l_q = self.qpts.cryst  # list of q-points in crystal coords
        epsilon_cutoff = epsinp.epsilon_cutoff

        self.sorted_gpts = [
            (self.l_gq[i_q].cryst, self.l_gq[i_q].norm2)
            for i_q in range(self.qpts.numq)
        ]

        # occupation numbers ------------------------------------------------

        occ_in = np.array(occ_in)

        occ = occ_in[:, 0:number_bands]
        # indices for ref: [index of kpoint, band index]
        l_i_v = np.where(occ == 1)  # list of indices of occupied   bands
        l_i_c = np.where(occ == 0)  # list of indices of unoccupied bands

        size = self.gspace.grid_shape
        n123 = np.prod(size)

        n_v_max = max(l_i_v[1]) + 1
        n_c_beg = min(l_i_c[1])
        n_c_max = number_bands - min(l_i_c[1])

        # Assume q=0 is at index 0 in l_q
        assert np.linalg.norm(l_q[0]) < TOLERANCE

        if bra_all_bands:
            l_i_sv_tuple = np.meshgrid(
                list(range(self.sigmainp.number_bands)), list(range(len(occ)))
            )
            l_i_sv = np.array([l_i_sv_tuple[1].flatten(),
                              l_i_sv_tuple[0].flatten()])
            l_i_v_m = l_i_sv
            n_v_m = number_bands
        else:
            l_i_v_m = l_i_v
            n_v_m = n_v_max

        if ket_all_bands:
            l_i_sc_tuple = np.meshgrid(
                list(range(self.sigmainp.number_bands)), list(range(len(occ)))
            )
            l_i_sc = np.array([l_i_sc_tuple[1].flatten(),
                              l_i_sc_tuple[0].flatten()])
            l_i_c_m = l_i_sc
            n_c_m = number_bands
            i_b_c_beg = 0  # Index where conduction bands begin
        else:
            l_i_c_m = l_i_c
            n_c_m = n_c_max
            i_b_c_beg = n_c_beg

        M = np.zeros((self.n_kpts, n_v_m, n_c_m, len(l_g[0])), dtype=complex)

        # UMKLAPP ----------------------------

        # Cycle to peridic image if k-q lies outside k grid (in crystal coords).
        # umklapp is the Correction that is to be added to bring the vector to 1st +BZ

        # umklapp = np.zeros_like(l_k)
        # umklapp = trunc_bril_legacy_vec(l_k)
        # FIXME: The following is redundant; l_kplusq IS l_k, because q=0 and umklapp=0.
        #        Just replace all l_kplusq with l_k.
        umklapp = -np.floor(np.around(l_k, 5))
        l_kplusq = l_k + umklapp
        assert np.allclose(l_k, l_kplusq)  # For debugging, to be removed

        # For efficiency, caching the valence band ifft's to improve serial performance,
        # but this should be removed later, or a cleaner solution must be found.
        @lru_cache(maxsize=int(n_v_max))
        def get_evc_gk_r2g(i_k_v, i_b_v):
            wfn_v = self.l_wfn[i_k_v]
            return wfn_v.gkspc.fft_mod.g2r(wfn_v.evc_gk[0, i_b_v, :])

        # MATRIX CALCULATION -------------------

        prev_i_k_ket = None  # To avoid recalculation for the same value of k
        for i_c in range(n_kpts * (n_c_m - i_b_c_beg)):
            i_k_ket = l_i_c_m[0][i_c]  # unoccupied k indices, repeated
            i_b_ket = l_i_c_m[1][i_c]  # unoccupied band indices, repeated

            if i_k_ket not in self.l_k_indices:
                continue

            i_k_c_insigma = np.where(self.l_k_indices == i_k_ket)[0][0]

            # phi_c = self.l_wfn[i_k_c].g2r(
            #     ([0], [i_b_c])
            # )  # Fourier transform to real space

            wfn_ket = self.l_wfn[i_k_ket]
            phi_ket = wfn_ket.gkspc.fft_mod.g2r(
                wfn_ket.evc_gk[0, i_b_ket, :]) / n123

            if prev_i_k_ket != i_k_ket:  # 'if' to avoid re-calculation
                prev_i_k_ket = i_k_ket

                # i_g = gmap_inv(l_g, N=np.array(size)[:,None])
                i_g = tuple(
                    np.mod(l_g, np.array(self.gspace.grid_shape)
                           [:, None]).astype(int)
                )

                umklapped_fft_driver = get_fft_driver()(
                    self.gspace.grid_shape,
                    i_g,
                    normalise_idft=False,
                    # NOTE: normalise_idft=False will be the default for all gw code
                )

            # k_c plus q for the cuurent i_k_c
            kcplusq = l_kplusq[i_k_ket]

            # obtain a list of indices of valence kpoints that match  k_c + q
            # In this case, q=0, which is at 0 index in the valence (bra) list
            # [ k-index , component index: 0,1,2 ]
            l_k_bra = l_k[l_i_v_m[0][:], :]

            l_i_match = np.nonzero(
                np.all(np.isclose(
                    l_k_bra, kcplusq[None, :], atol=1e-5), axis=1)
            )[0]

            # for k_v == k_c + q:
            for i_v in l_i_match:
                i_k_bra = l_i_v_m[0][i_v]  # occupied k indices, repeated
                i_b_bra = l_i_v_m[1][i_v]  # occupied band indices

                # phi_bra = self.l_wfn[i_k_bra].g2r(([0], [i_b_bra]))[0]
                # prod = np.multiply(np.conj(phi_ket), phi_bra)
                # fft_prod = self.l_gq[0].fft_dri.r2g(prod, idxgrid=tuple(i_g))

                phi_bra = get_evc_gk_r2g(i_k_bra, i_b_bra) / n123
                prod = np.multiply(np.conj(phi_ket), phi_bra)
                fft_prod = umklapped_fft_driver.r2g(prod)

                M[
                    i_k_c_insigma, i_b_bra, i_b_ket - i_b_c_beg
                ] = fft_prod  # / sqrt_Ec_Ev

        for i, g in enumerate(l_g.T):
            mag_g = self.gspace.recilat.norm(g) ** 2
            if mag_g > epsilon_cutoff:
                M[:, :, :, i] *= 0

        return M * n123

    # ==================================================================
    # Static Sigma methods

    def sigma_x(self, yielding=True):
        """
        Fock exchange energy term
        =========================
        Returns Sigma_x[i_k, i_band] for diag = True

        - Sum _(n" over occupied, q, G, G')   M^*_n"n'(k,-q,-G)  M_n"n'(k,-q,-G')  delta_GG'  v(q+G')
        - Sum _(n" over occupied, q, G=G')    M^*_n"n'(k,-q,-G)  M_n"n'(k,-q,-G)  v(q+G)
        """

        # Init empty sigma matrix
        # number_bands = self.sigmainp.number_bands
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max-band_index_min+1
        sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)

        # n   : k-points
        # i   : valence    states
        # j,l : conduction states
        # k   : G
        # m   : G'
        # in sigma_x the delta_GG' means k=m

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0
        # epsilon_cutoff = sigmainp.epsilon_cutoff

        gpts = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]


        # Embarassingly parallel, just use microprocessing library
        def compute_sigma_q(i_q):
            # Get qpt value
            qvec = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            minus_qvec = l_q[i_minusq]

            # Calculate map from G to -G : map_g_to_minusg
            g = gpts[i_q].T

            # map from G to -G
            g_minusq = gpts[i_minusq].T
            g = gpts[i_q].T
            g_to_mg = self.map_g_to_g_minusq(g, g_minusq, qvec, minus_qvec)

            vqg = self.vcoul.vcoul[i_q]
            
            sigma_q = np.zeros_like(sigma)
            
            # yielding: To reduce memory consumption
            if yielding:
                einstr = "njm,njm,m->nj"
                for M in self.matrix_elements(i_minusq, ket_all_bands=True, yielding=yielding):
                    M = M[..., g_to_mg]
                    sigma_q += np.einsum(einstr, np.conj(M), M, vqg, optimize=True)

            else:
                einstr = "nijm,nijm,m->nj"
                M = next(self.matrix_elements(
                    i_minusq, ket_all_bands=True, yielding=yielding))[..., g_to_mg]
                sigma_q = np.einsum(einstr, np.conj(M), M, vqg, optimize=True)
            
            return sigma_q

        for i_q in trange(len(l_q), desc="Sigma_X"):
            sigma_q = compute_sigma_q(i_q)
            sigma -= sigma_q

        return sigma

    def sigma_sx_static(self):
        """
        Static Screened Exchange
        ========================
        """

        # Setting einstein summation string for M* M epsinv v

        # Init empty sigma matrix
        number_bands_inner = self.sigmainp.number_bands
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max-band_index_min+1
        # sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)

        sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
        einstr = "nijk,nijm,mk,m->nj"

        # if diag:
        #     sigma = np.zeros((self.n_kpts, number_bands), dtype=complex)
        #     einstr = "nijk,nijm,mk,m->nj"
        # else:
        #     sigma = np.zeros(
        #         (self.kpts.numk, number_bands, number_bands), dtype=complex
        #     )
        #     einstr = "nijm,nilm,m->njl"

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0
        # epsilon_cutoff = sigmainp.epsilon_cutoff

        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]
        # print("gpts\n", np.array(g_cryst[0]))

        # i   : valence    states
        # j,l : conduction states
        # k   : G
        # m   : G'
        # in sigma_x the delta_GG' means k=m


        # print("Sx Beginning iq loop")
        for i_q in trange(len(l_q), desc="Sigma_SX_Static"):
            # Get qpt value
            qpt = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            mqpt = l_q[i_minusq]

            # Calculate map from G to -G : map_g_to_minusg
            g = g_cryst[i_q].T

            # map from G to -G
            g_minusq = g_cryst[i_minusq].T
            g = g_cryst[i_q].T
            g_to_mg = self.map_g_to_g_minusq(g, g_minusq, qpt, mqpt)

            epsinv_I = self.l_epsinv_I[i_q]

            vqg = self.vcoul.vcoul[i_q]

            # Calculate matrix elements
            M = next(self.matrix_elements(i_minusq, ket_all_bands=True))

            # debug('M.shape')#,'M')
            sigma_q = np.einsum(
                einstr,
                np.conj(M)[..., g_to_mg],
                M[..., g_to_mg],
                epsinv_I,
                vqg,
                optimize=True,
            )

            # print(f"i_q:{i_q}")
            # print("sigma_q"); print(np.real(np.diag(sigma_q)))

            sigma += sigma_q
            # i   : valence    states
            # j,l : conduction states
            # k   : G
            # m   : G'
        return -1 * sigma

    def sigma_ch_static(self, diag=True):
        """
        Static Coulomb Hole (partial sum)
        =================================
        """
        # I think the error is due to the difference in ordering g-vecs in Matrix elements
        # >> Tested: The ordering is same, so the problem is elsewhere.

        # Setting einstein summation string for M* M epsinv v
        # Init empty sigma matrix
        number_bands_inner = self.sigmainp.number_bands
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max-band_index_min+1

        if diag:
            sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
            einstr = "nijk,nijm,mk,m->nj"
        else:
            sigma = np.zeros(
                (self.kpts.numk, number_bands_outer, number_bands_outer), dtype=complex
            )
            einstr = "nijm,nilm,m->njl"

        # n   : k-points
        # i   : valence    states
        # j,l : conduction states
        # k   : G
        # m   : G'

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0
        # epsilon_cutoff = sigmainp.epsilon_cutoff

        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]
        # print("gpts\n", np.array(g_cryst[0]))

        # w0 = w_sph()

        # --------------------------------------------------
        # Debugging vcoul: Check with BGW values for vqg

        for i_q in trange(
            self.q_loop_beg, self.q_loop_end, desc="Sigma_CH_Static_Partial"
        ):
            # Get qpt value
            qpt = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            mqpt = l_q[i_minusq]

            # Calculate map from G to -G : map_g_to_minusg
            g = g_cryst[i_q].T

            # map from G to -G
            g_minusq = g_cryst[i_minusq].T
            g = g_cryst[i_q].T
            g_to_mg = self.map_g_to_g_minusq(g, g_minusq, qpt, mqpt)

            epsinv_I = self.l_epsinv_I[i_q]

            # Replace vcoul with BGW vcoul
            # if test_vcoul_from_bgw:
            #     vqg = vcoul_bgw[cumsum_ngk[i_q] : cumsum_ngk[i_q] + ngg[i_q]]
            # else:

            # Get ususal Coulomb potential: vqg = vcoulomb(g)
            vqg = self.vcoul.vcoul[i_q]

            # Calculate matrix elements
            M = next(self.matrix_elements(
                i_minusq, ket_all_bands=True, bra_all_bands=True))

            sigma_q = np.einsum(
                einstr,
                np.conj(M)[..., g_to_mg],
                M[..., g_to_mg],
                epsinv_I,
                vqg,
                optimize=True,
            )

            sigma += sigma_q

            # i   : valence    states
            # j,l : conduction states
            # k   : G
            # m   : G'
        return 0.5 * sigma

    def sigma_ch_exact_new(self, diag=True, test_vcoul_from_bgw=global_test_vcoul):
        """
        Static Coulomb Hole (Exact)
        ========================
        0.5 * \Sum_{q,G,G'} = M_{n,n'}(k, q=0, G'-G) * [\eps^{-1}_{G,G'}(q;0) - \delta_{G,G'}] * v(q+G')

        1e-3 disagreement: Doubt goes to limits on G'-G: is it all G and G' within cutoff or G-G' within cutoff
        """

        # Init empty sigma matrix
        # number_bands = self.sigmainp.number_bands
        number_bands_inner = self.sigmainp.number_bands
        
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max-band_index_min+1

        # Setting einstein summation string for M* M epsinv v

        if diag:
            sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
            einstr = "kijjm,mk,m->ij"
            # note km -> mk for bgw epsmat
        else:
            sigma = np.zeros(
                (self.kpts.numk, number_bands_outer, number_bands_outer), dtype=complex
            )
            einstr = "kijlm,km,m->ijl"

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0
        # epsilon_cutoff = sigmainp.epsilon_cutoff

        sorted_g = [
            (self.l_gq[i_q].cryst, self.l_gq[i_q].norm2)
            for i_q in range(self.qpts.numq)
        ]
        # w0 = w_sph()

        # --------------------------------------------------
        # Debugging vcoul: Check with BGW values for vqg

        if test_vcoul_from_bgw:
            vcoul_bgw = np.array(read_txt(dirname + "vcoul"))

            ngg = []
            for i_q in range(len(l_q)):
                ngg.append(len(sorted_g[i_q][0].T))
            ngg = np.array(ngg)

            cumsum_ngk = np.cumsum(ngg)
            cumsum_ngk = np.roll(cumsum_ngk, 1)
            cumsum_ngk[0] = 0

        # Assume received epsinv and matrix elements ordered by l_q
        for i_q in trange(len(l_q), desc="Sigma_CH_Static_Exact"):
            # Get qpt value
            # qpt = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            # mqpt = l_q[i_minusq]

            # Calculate map from G to -G : map_g_to_minusg
            g = sorted_g[i_q][0].T

            # map from G to -G
            # g_minusq = sorted_g[i_minusq][0].T
            g = sorted_g[i_q][0].T
            # g_to_mg = self.map_g_to_g_minusq(g,g_minusq,qpt,mqpt)

            # Get epsinv-I
            epsinv_I = self.l_epsinv_I[i_q]

            # Calculate/Load Coulomb potential
            vqg = self.vcoul.vcoul[i_q]

            # Replace vcoul with BGW vcoul
            if test_vcoul_from_bgw:
                vqg = vcoul_bgw[cumsum_ngk[i_q]: cumsum_ngk[i_q] + ngg[i_q]]

            # G'-G = gpmg = Gprime minus G
            gpmg = np.array([[gpvec - gvec for gpvec in g] for gvec in g])

            # Calculate matrix elements
            M = self.matrix_elements_sigma_exact(
                np.vstack(gpmg).T, ket_all_bands=True, bra_all_bands=True
            )
            # debug('M.shape')

            # Reshape M from all possible (G'-G) to (G',G).
            # We had used vstack with gpmg to create a list of g-vectors
            # to be sent to Matrix_elemts_augmented function.
            # Therefore, we need to unpack the result.
            # Ensure that the result does the same cutoff business as BGW with G'-G.
            # !For now, assuming nothing special is done to cutoff |G'-G|.

            len_g = len(g)
            M = np.array([M[..., len_g * i: len_g * (i + 1)]
                         for i in range(len_g)])
            # print(M.shape,M[0,0,0])

            sigma_q = 0.5 * np.einsum(einstr, M, epsinv_I, vqg, optimize=True)

            sigma += sigma_q
            # diag: einstr = "kijjm,mk,m->ij"
            # i   : valence    states
            # j,l : conduction states
            # k   : G
            # m   : G'
        return sigma

    # ==================================================================
    # GPP Sigma methods

    def sigma_gpp_omegas(self, qpt, g, vqg, epsinv_I):
        """GPP Omega Calculator

        (gsm) equations (17), (20), (21) from [PRB 40, 3162 (1989) (Zhang-Tomanek-Cohen-Louie-Hybertsen-1989)]

        Notes
        =====
        1 ! Compute address of g-gp, and if it is a vector for which
            we have the density, get rho(g-gp); if out of bounds,
            skip this g

        2 ! If I_epsggp is too small, then we skip this (G,G`) entry
            This only happens when eps is 1 on diagonal or 0 off diagonal
            but, this means no screening correct and is already treated properly in bare
            exchange term

        3 ! If Omega2 is too small, then we skip this (G,G`) entry
            JRD: I am not sure why we have to cycle here... :/ Probably not needed
            FHJ: If Omega2->0, both the SX and the CH terms vanish. But the CH term
            goes to zero as Omega2/wtilde ~ Omega2/sqrt(Omega2). Skipping this term is
            probably better than risking a 0/sqrt(0) division.

        4 ! In computing the sums in Eqs. (22) and (23) we drop terms in
            certain circumstances to save time and improve numerical precision.
            We neglect the terms for which
            |1_GG' - epsinv_GG'(q;0)|, or
            |λ_GG'(q)|, or
            |cos(φ_GG'(q))|
            are less than a given tolerance, since these terms have
            a vanishing contribution to the matrix elements of the self-energy.
            This avoids ill-conditioned limits due to some of the intermediate
            quantities here being undeﬁned.


        """
        # Read density from RHO binary Fortran file
        #! If g=q=0, we have to avoid dividing by zero; : NOT IMPLEMENTED

        # i_g0 : index of (0,0,0) vector in the gvecs list of rho
        # required for calculing rho(0)

        # RHO data
        rho = self.rho

        # vxc_data = inp.read_vxc(dirname+'vxc.dat')
        # epsinv_I = epsinv_I.T

        # i_g0 = self.find_indices([[0,0,0]], rho.gvecs.tolist())[0]
        i_g0 = i_gmgpvec = self.rho_dict[self.rhohash([0, 0, 0])]
        if i_g0 < 0:  # The difference does not exist in rho gvecs list
            raise ValueError("G=0 vector not found in G-vectors list of RHO.")
        nelec = rho.rho[i_g0]

        # define ω_p^2
        wp2 = self.ryd**2 * 16 * np.pi * nelec / self.crystal.reallat.cellvol

        # BGW Comments:
        # ============
        # wpg%nelec(ispin)=dble(wpg%rho(1,ispin))
        # wpg%wpsq(ispin)=ryd*ryd*16.0d0*PI_D*wpg%nelec(ispin)/crys%celvol
        # ./Epsilon/epsinvomega.f90:422:  wp2=ryd*ryd*16.0d0*PI_D*rho0/celvol
        # RYD = 13.60569253_dp

        # ! This is unacceptable because it means the number of electrons is negative,
        # ! and the plasma frequency will be imaginary!
        # if(any(wpg%nelec(1:kp%nspin) < TOL_Zero)) then
        #   write(0,*) wpg%nelec(:)
        #   call die("Charge density in RHO has negative part for G=0", only_root_writes = .true.)
        # endif

        # ! The special case q=g=0:
        # ! If gp=0, then wpmtx=wp**2 times a possible
        # ! (1-cos(0))=0 if truncating

        #     if (igpadd.eq.1) then
        #         wpmtx(ig) = fact*rho_g_minus_gp
        # !            if (sig%icutv/=TRUNC_NONE) wpmtx(ig) = 0.0d0

        # ! When q=g=0, gp<>0, the result is just set to zero
        # ! (it is infinite for true coulomb interaction and zero
        # ! if truncated).
        # ------------------------------------------------------------------------------------------

        # init matrices
        len_g = len(g)
        omega2 = np.zeros((len_g, len_g), dtype=complex)
        wtilde2 = np.zeros((len_g, len_g), dtype=complex)
        wpmtx = np.zeros((len_g, len_g), dtype=complex)

        # from read_rho_vxc.f90:
        # if(any(abs(aimag(wpg%rho(1,:))) > TOL_Zero)) then
        #   call die("Charge density in RHO has imaginary part for G=0", only_root_writes = .true.)
        # endif
        # rho.
        # self.gspace.

        # print("Calculating Omegas: G-loops 1")
        for i_gp, gpvec in enumerate(g):
            for i_g, gvec in enumerate(g):
                # i_gmgpvec : index of G-G'
                # i_gmgpvec = self.find_indices([(gvec-gpvec).tolist()], rho.gvecs.tolist())[0]
                # i_gmgpvec = np.searchsorted(self.rho_gvecs_hashed, v = self.rhohash(gvec-gpvec), sorter=self.rho_sorter)
                i_gmgpvec = self.rho_dict[self.rhohash(gvec - gpvec)]
                if i_gmgpvec < 0:  # The difference does not exist in rho gvecs list
                    continue  # In accordance with BGW. See #1

                # Correct rho
                rho_gmgpvec = rho.rho[i_gmgpvec].copy()
                # or np.real(rho_gmgpvec) < 0:
                if np.imag(rho_gmgpvec) > TOLERANCE:
                    rho_gmgpvec *= 0

                # Skip small I-epsinv
                # debug('epsinv_I[i_g,i_gp]')
                if abs(epsinv_I[i_g, i_gp]) < TOLERANCE:
                    continue  # See #2

                    # / rho.rho[i_g0] )
                omega2[i_g, i_gp] = (
                    wp2
                    * (rho_gmgpvec / nelec)
                    * np.einsum(
                        "j,jk,k", gvec + qpt, self.gspace.recilat.metric, gpvec + qpt
                    )
                    * vqg[i_g]
                    / (8 * np.pi)
                )

                cycle = False
                if i_g == 0 and TOLERANCE > abs(
                    np.einsum(
                        "j,jk,k", gvec + qpt, self.gspace.recilat.metric, gvec + qpt
                    )
                ):
                    if i_gp != 0:
                        cycle = True
                    else:
                        omega2[i_g, i_gp] = wp2 * rho_gmgpvec / nelec

                # if (abs(Omega2).lt.TOL_Small) cycle
                if np.abs(omega2[i_g, i_gp]) < TOLERANCE:
                    omega2[i_g, i_gp] *= 0

                # wtilde2_temp = Omega2 / I_epsggp
                wtilde2_temp = omega2[i_g, i_gp] / (-epsinv_I[i_g, i_gp])

                # lambda = abs(wtilde2_temp)
                lambda_abs = np.absolute(wtilde2_temp)

                # Skip small lambda
                # if (lambda .lt. TOL_Small) cycle
                if abs(lambda_abs) < TOLERANCE:
                    continue  # In accordance with BGW, See #4

                # phi = atan2(IMAG(wtilde2_temp), dble(wtilde2_temp))
                phi = np.arctan2(np.imag(wtilde2_temp), np.real(wtilde2_temp))

                # Skip small cos(phi)
                # if (abs(cos(phi)) .lt. TOL_Small) cycle
                if abs(np.cos(phi)) < TOLERANCE:
                    continue  # In accordance with BGW, See #4

                wtilde2[i_g, i_gp] = lambda_abs / np.cos(phi)
                wpmtx[i_g, i_gp] = omega2[i_g, i_gp]
                omega2[i_g, i_gp] *= 1 - 1j * np.tan(phi)
                # wtilde2[i_g, i_gp]  *= (1-1j*np.tan(phi))

                # if False:
                # print(i_g, i_gp, end = "\t")
                # print("Omega2, wtilde2", omega2[i_g, i_gp], wtilde2[i_g, i_gp], (-1*epsinv_I[i_g, i_gp]))

        # print("Calculating Omegas: G-loops 2")
        for i_gp, gpvec in enumerate(g):
            for i_g, gvec in enumerate(g):
                if wtilde2[i_g, i_gp] < 0:
                    # Square of plasma freq < 0: unphysical, see paper for why
                    # print("w~ < 0")
                    wtilde2[i_g, i_gp] = 1e24

                    # Omega2 = wtilde2 * I_eps_array(ig,my_igp)
                    omega2[i_g, i_gp] = wtilde2[i_g, i_gp] * \
                        (-1) * epsinv_I[i_g, i_gp]

        assert not np.isnan(wtilde2).any()
        # print("Omega complete")

        return omega2, wtilde2

    def sigma_sx_gpp(self, dE=0, diag=True):
        """
        (H.L.) Plasmon Pole Screened Exchange
        ======================================

                                            Omega^2(G,G`)
        SX(E) = M(n,G)*conj(M(m,G`)) * ------------------------ * Vcoul(G`)
                                    (E-E_n1(k-q))^2-wtilde^2
        """

        # Setting einstein summation string for M* M epsinv v
        # number_bands = self.sigmainp.number_bands
        
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max-band_index_min+1

        if diag:
            sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
            # einstr = "kvcg, kvcp, gp, kvcgp, p -> kc"
            einstr = "kvcg, kvcp, kvcgp, p -> kc"
            #         M*    M     ssx    V    Σ_n=n',c
            # denom = 1/((Ev-Ec)^2 - ω~^2)
        else:
            raise NotImplementedError

        # k   : k-point
        # v   : valence    states (bands)
        # c,d : conduction states (bands)
        # g   : G
        # p   : G'

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        # set q0 = 0
        l_q[self.qpts.index_q0] *= 0
        # epsilon_cutoff = sigmainp.epsilon_cutoff

        # sorted_g = [
        #     (self.l_gq[i_q].cryst, self.l_gq[i_q]._norm2)
        #     for i_q in range(self.qpts.numq)
        # ]
        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]

        # --------------------------------------------------

        # print("SX GPP: Beginning iq loop")
        for i_q in trange(0,len(l_q), desc="Sigma_SX_GPP"):
            # Get qpt value
            qpt = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            mqpt = l_q[i_minusq]

            # map from G to -G
            g_minusq = g_cryst[i_minusq].T
            g = g_cryst[i_q].T
            g_to_mg = self.map_g_to_g_minusq(g, g_minusq, qpt, mqpt)

            # Load epsinv-I
            epsinv_I = self.l_epsinv_I[i_q].T

            # Load vqg
            vqg = self.vcoul.vcoul[i_q]

            # print("Calculating M")
            # Calculate matrix elements; also returns mean field energy eigenvals for ket states
            M, E_bra, E_ket = next(self.matrix_elements(
                i_minusq, ket_all_bands=True, ret_E=True
            ))
            # print("M", M)
            # print("E_ket", E_ket)
            # print("E_bra", E_bra)
            # exit()


            # print("Calculating Omegas")
            # Calculate Omega expression
            # See notes #1 and #2
            omega2, wtilde2 = self.sigma_gpp_omegas(qpt, g, vqg, epsinv_I)
            # print(omega2)


            # print("Omega2", omega2[2])
            # print("wtilde2", wtilde2[3])
            # exit()
            # Calculate (E_nk-E)^2
            # denominator_E2          = np.square((Eqp_0/2-Emf))[:,:4] # 0.07349865* # np.square((Eqp_0-Emf)/sigma_factor) #
            # E = E_ket + dE

            # print("Calculating E")
            E = E_ket + dE
            # print(i_q,"E",E)
            # print((E_ket+dE)[self.slice_l_k])
            wxt = self.ryd * (
                E[:, np.newaxis, :] - E_bra[:, :, np.newaxis]
            )  # [self.slice_l_k]
            # print(E_ket)
            # print(E_bra)

            denominator_E2 = np.square(wxt)
            # Calculate (E_nk-E)^2 - wtilde^2
            denominator = np.subtract.outer(denominator_E2, wtilde2)

            # wdiff = wxt - wtilde        ! E - E_n"kq - wtilde
            # cden = wdiff                ! complex denominator
            # rden = cden * CONJG(cden)   ! squared (real) denominator
            # rden = 1D0 / rden           ! reciprocated denominator
            # delw = wtilde * CONJG(cden) * rden
            # delwr = delw*CONJG(delw)
            # wdiffr = wdiff*CONJG(wdiff)

            # print("Calculating wtilde")

            wtilde = np.sqrt(wtilde2)

            wdiff = np.subtract.outer(wxt, wtilde)
            cden = wdiff
            rden = np.square(cden)
            rden[np.where(rden == 0)] = np.finfo(float).eps
            rden = np.reciprocal(rden)
            delw = wtilde * np.conj(cden) * rden
            delwr = delw * np.conj(delw)
            wdiffr = wdiff * np.conj(wdiff)

            condbrancha = np.logical_and(
                wdiffr > self.limittwo, delwr < self.limitone)
            condbranchb = delwr > TOLERANCE
            # wherebranchnota  = np.where(np.logical_not(condbrancha))
            # wherebranchbnota = np.where(
            #     np.logical_and(np.logical_not(condbrancha), condbranchb)
            # )
            wherebranchnotanotb = np.where(
                np.logical_and(np.logical_not(condbrancha),
                               np.logical_not(condbranchb))
            )

            mask_branchbnota = np.logical_and(
                np.logical_not(condbrancha), condbranchb)

            # branch A
            # sch = delw * I_eps_array(ig,my_igp)
            # cden = wxt**2 - wtilde2
            # rden = cden*CONJG(cden)
            # rden = 1D0 / rden
            # ssx = Omega2 * CONJG(cden) * rden
            cden = denominator
            rden = np.square(cden)
            rden[np.where(rden == 0)] = np.finfo(float).eps
            rden = np.reciprocal(rden)

            # ssx = np.einsum("lm, ijklm, ijklm-> ijklm", omega2, np.conj(cden), rden)
            ssx = np.einsum(
                "gp, kvcgp, kvcgp -> kvcgp", omega2, np.conj(cden), rden, optimize=True
            )  # there was conj(rden) for some reason
            ssx = np.nan_to_num(ssx)

            # branch B
            # else if ( delwr .gt. TOL_Zero) then
            #       print *, "=> BRANCH B: delwr .gt. TOL_Zero, delwr", delwr
            #       sch = 0.0d0
            #       cden = (4.0d0 * wtilde2 * (delw + 0.5D0 ))
            #       rden = cden*MYCONJG(cden)
            #       rden = 1D0 / rden
            #       ssx = -Omega2 * MYCONJG(cden) * rden * delw

            cden_branchbnota = 4 * np.einsum(
                "gp, kvcgp -> kvcgp", wtilde2, (delw + 0.5)
            )
            cden = np.where(mask_branchbnota, cden_branchbnota, cden)

            rden_branchbnota = np.einsum(
                "kvcgp, kvcgp -> kvcgp", cden, np.conj(cden))
            rden_branchbnota[np.where(
                rden_branchbnota == 0)] = np.finfo(float).eps
            rden_branchbnota = np.reciprocal(rden_branchbnota)

            rden = np.where(mask_branchbnota, rden_branchbnota, rden)

            # print("Calculating ssx")

            ssx_branchbnota = -1 * np.einsum(
                "gp, kvcgp, kvcgp, kvcgp -> kvcgp",
                omega2,
                np.conj(cden),
                rden,
                delw,
                optimize=True,
            )
            # pprint_arr_2d(ssx_branchbnota[0,0,1])

            ssx = np.where(
                mask_branchbnota,
                ssx_branchbnota,
                ssx,
            )

            ssx[wherebranchnotanotb] *= 0

            if False:
                test_preview_slice_count = number_bands_outer
                # print("sch", sch[0,0,0,0,:10])
                print("ssx", ssx[0, 0, 0, 0, :test_preview_slice_count])
                print("cden", cden[0, 0, 0, 0, :test_preview_slice_count])
                print("rden", rden[0, 0, 0, 0, :test_preview_slice_count])
                print("delw", delw[0, 0, 0, 0, :test_preview_slice_count])

            # ssxcutoff = sig%sexcut*abs(I_eps_array(ig,my_igp))
            # BGW ref sigma.inp: gpp_sexcutoff [float]
            # Cutoff for the poles in SX within GPP. Divergent contributions that are supposed to sum to zero are removed. This is dimensionless, the default value is 4.0
            # if (abs(ssx) .gt. ssxcutoff .and. wxt .lt. 0.0d0) then
            #     print *, "abs(ssx) .gt. ssxcutoff .and. wxt .lt. 0.0d0", ssxcutoff
            #     ssx=0.0d0
            # endif

            # Screened exchange cutoff
            np.place(
                ssx,
                np.logical_and(
                    abs(ssx) > 4 * np.abs(epsinv_I), (wxt <
                                                      0)[:, :, :, None, None]
                ),
                [0],
            )

            # print("Calculating Sigma_q")
            # print("cden",cden[1,2,3])
            # print("rden",rden[2,3,1])
            # print("delw",delw[1,3,2])
            # exit()

            sigma_q = np.einsum(
                einstr,
                np.conj(M)[..., g_to_mg],
                M[..., g_to_mg],
                ssx,
                vqg,
                optimize=True,
            )
            assert not np.isnan(sigma_q).any()

            sigma += sigma_q

        return -sigma

    def sigma_ch_gpp(self, dE=0, diag=True):
        """
        Plasmon Pole Coulomb Hole (partial sum)
        =======================================

                                            Omega^2(G,G`)
        CH(E) = M(n,G)*conj(M(m,G`)) * ----------------------------- * Vcoul(G`)
                                    2*wtilde*[E-E_n1(k-q)-wtilde]

        """

        # Setting einstein summation string for M* M epsinv v

        # number_bands = self.sigmainp.number_bands
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max-band_index_min+1
        print("dE:", dE)

        if diag:
            sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
            # einstr = "kvcg, kvcp, gp, kvcgp, p -> kc"
            einstr = "kvcg, kvcp, kvcgp, p -> kc"
            #         M*    M     ssx    V    Σ_n=n',c
            # denom = 1/((Ev-Ec)^2 - ω~^2)
        else:
            raise NotImplementedError

        # k   : k-point
        # v   : valence    states (bands)
        # c,d : conduction states (bands)
        # g   : G
        # p   : G'

        # w0 = w_sph()

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0
        # epsilon_cutoff = sigmainp.epsilon_cutoff

        # sorted_g = [
        #     (self.l_gq[i_q].cryst, self.l_gq[i_q]._norm2)
        #     for i_q in range(self.qpts.numq)
        # ]
        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]

        for i_q in trange(
            self.q_loop_beg, self.q_loop_end, desc="Sigma_CH_GPP_Partial"
        ):
            # Get qpt value
            qpt = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            mqpt = l_q[i_minusq]

            # map from G to -G
            g_minusq = g_cryst[i_minusq].T
            g = g_cryst[i_q].T
            g_to_mg = self.map_g_to_g_minusq(g, g_minusq, qpt, mqpt)

            # Load epsinv-I
            epsinv_I = self.l_epsinv_I[i_q].T

            vqg = self.vcoul.vcoul[i_q]

            # print("Calculating M")
            # Calculate matrix elements; also returns mean field energy eigenvals for ket states
            M, E_bra, E_ket = next(self.matrix_elements(
                i_minusq, ket_all_bands=True, bra_all_bands=True, ret_E=True
            ))  # , row_all_bands=True

            # Calculate Omega expression
            # See #1 and #2
            # print("Calculating Omegas")
            omega2, wtilde2 = self.sigma_gpp_omegas(qpt, g, vqg, epsinv_I)

            # ----------------------- FOCUS HERE --------------------------

            # Calculate (E_nk-E)^2
            # denominator_E2          = np.square((Eqp_0/2-Emf))[:,:4] # 0.07349865* # np.square((Eqp_0-Emf)/sigma_factor) #
            # print(E.shape)
            E = E_ket + dE

            # print((E_ket+dE)[self.slice_l_k])
            wxt = self.ryd * (E[:, np.newaxis, :] - E_bra[:, :, np.newaxis])

            denominator_E2 = np.square(wxt)
            # Calculate (E_nk-E)^2 - wtilde^2
            denominator = np.subtract.outer(denominator_E2, wtilde2)

            # print("denominator.shape:", denominator.shape)

            # wdiff = wxt - wtilde        ! E - E_n"kq - wtilde
            # cden = wdiff                ! complex denominator
            # rden = cden * CONJG(cden)   ! squared (real) denominator
            # rden = 1D0 / rden           ! reciprocated denominator
            # delw = wtilde * CONJG(cden) * rden
            # delwr = delw*CONJG(delw)
            # wdiffr = wdiff*CONJG(wdiff)

            wtilde = np.sqrt(wtilde2)

            wdiff = np.subtract.outer(wxt, wtilde)
            cden = wdiff
            rden = np.square(cden)
            rden[np.where(rden == 0)] = np.finfo(float).eps
            rden = np.reciprocal(rden)
            delw = wtilde * np.conj(cden) * rden
            delwr = delw * np.conj(delw)
            wdiffr = wdiff * np.conj(wdiff)

            # TODO: Check and comment what branch A and B denote
            condbrancha = np.logical_and(
                wdiffr > self.limittwo, delwr < self.limitone)
            condbranchb = delwr > TOLERANCE
            # wherebranchnota  = np.where(np.logical_not(condbrancha))
            wherebranchbnota = np.where(
                np.logical_and(np.logical_not(condbrancha), condbranchb)
            )
            wherebranchnotanotb = np.where(
                np.logical_and(np.logical_not(condbrancha),
                               np.logical_not(condbranchb))
            )

            # TODO: Check and comment what branch A and B denote
            # branch A
            # sch = delw * I_eps_array(ig,my_igp)
            # cden = wxt**2 - wtilde2
            # rden = cden*CONJG(cden)
            # rden = 1D0 / rden
            # ssx = Omega2 * CONJG(cden) * rden
            cden = denominator
            rden = np.square(cden)
            rden[np.where(rden == 0)] = np.finfo(float).eps
            rden = np.reciprocal(rden)

            sch = np.einsum(
                "kvcgp, gp -> kvcgp", delw, (-1) * epsinv_I, optimize=True
            )  # there was conj(rden) for some reason
            sch = np.nan_to_num(sch)

            # branch B
            # else if ( delwr .gt. TOL_Zero) then
            #       print *, "=> BRANCH B: delwr .gt. TOL_Zero, delwr", delwr
            #       sch = 0.0d0
            #       cden = (4.0d0 * wtilde2 * (delw + 0.5D0 ))
            #       rden = cden*MYCONJG(cden)
            #       rden = 1D0 / rden
            #       ssx = -Omega2 * MYCONJG(cden) * rden * delw

            cden[wherebranchbnota] = (
                4
                * np.einsum("gp, kvcgp -> kvcgp", wtilde2, (delw + 0.5))[
                    wherebranchbnota
                ]
            )
            rden[wherebranchbnota] = np.einsum(
                "kvcgp, kvcgp -> kvcgp", cden, np.conj(delw)
            )[wherebranchbnota]

            rden[wherebranchbnota] = np.reciprocal(rden[wherebranchbnota])

            sch[wherebranchbnota] *= 0
            # (-1*np.einsum( "gp, kvcgp, kvcgp, kvcgp -> kvcgp", omega2, np.conj(cden), rden, delw, optimize=True))[ wherebranchbnota ]

            sch[wherebranchnotanotb] *= 0

            sigma_q = np.einsum(
                einstr,
                np.conj(M)[..., g_to_mg],
                M[..., g_to_mg],
                sch,
                vqg,
                optimize=True,
            )  # [self.slice_l_k]
            assert not np.isnan(sigma_q).any()

            sigma += sigma_q

        return 0.5 * sigma

    # ==================================================================
    # Methods to run full calculations

    def calculate_static_cohsex(self):
        # number_bands = self.sigmainp.number_bands
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        # number_bands_outer = band_index_max-band_index_min+1

        # Calculate on-shell QP energy = Emf - Vxc + Sig(Eo)
        # Vxc data
        vxc_data = inp.read_vxc(dirname + "vxc.dat")
        vxc = np.array(vxc_data.vxc)[self.slice_l_k, band_index_min-1:band_index_max]

        # Emf data
        
        emf_factor = 1/ELECTRONVOLT_RYD

        emf = np.array([self.l_wfn[i_k].evl for i_k in self.l_k_indices])
        emf = np.array(emf[:,0,band_index_min-1:band_index_max]) * emf_factor
        print("emf\n",emf)
        

        sigma_x_mat = self.sigma_x()

        sigma_sx_mat = self.sigma_sx_static()

        sigma_ch_mat = self.sigma_ch_static()

        sigma_ch_exact_mat = self.sigma_ch_exact_new()

        sigma_mat = sigma_sx_mat + sigma_ch_exact_mat + sigma_x_mat
        sigma_mat = np.around((sigma_mat) * Sigma.sigma_factor, 6)
        print("Sig (ExactCH):")
        print(sigma_mat.T)
        print("Eqp0 (ExactCH):")
        print((sigma_mat + emf - vxc).T)


        sigma_mat = sigma_sx_mat + sigma_ch_mat + sigma_x_mat
        sigma_mat = np.around((sigma_mat) * Sigma.sigma_factor, 6)
        print("Sig (PartialCH):")
        print(sigma_mat.T)
        print("Eqp0 (PartialCH):")
        print((sigma_mat + emf - vxc).T)

    def calculate_gpp(self):
        # number_bands = self.sigmainp.number_bands
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        # number_bands_outer = band_index_max-band_index_min+1

        # Calculate on-shell QP energy = Emf - Vxc + Sig(Eo)
        # Vxc data
        vxc_data = inp.read_vxc(dirname + "vxc.dat")
        vxc = np.array(vxc_data.vxc)[self.slice_l_k][:, band_index_min-1:band_index_max]

        # Emf data
        
        emf_factor = 1/ELECTRONVOLT_RYD

        emf = np.array([self.l_wfn[i_k].evl for i_k in self.l_k_indices])
        emf = np.array(emf[:,0,band_index_min-1:band_index_max])* emf_factor
        print("emf\n",emf)

        # before slicing emf, create dE with the correct shape
        dE = np.zeros_like(emf)
        # print(dE)

        # print("emf", emf.shape, emf)

        # Calculate Eqp0
        # ==============

        sigma_sx_gpp_mat = self.sigma_sx_gpp()
        print("Sigma SX GPP")
        print(np.around((sigma_sx_gpp_mat) * Sigma.sigma_factor, 6))
        print()
        # print(sigma_sx_gpp_mat)
        # print()

        sigma_ch_gpp_mat = self.sigma_ch_gpp()
        print("Sigma CH GPP")
        print(np.around((sigma_ch_gpp_mat) * Sigma.sigma_factor, 6))
        print()
        # print(sigma_ch_gpp_mat)
        # print()

        sigma_ch_static_mat = self.sigma_ch_static()
        print("Sigma CH STATIC COHSEX")
        print(np.around((sigma_ch_static_mat) * Sigma.sigma_factor, 6))
        print()
        # print(sigma_ch_static_mat)
        # print()

        sigma_ch_exact_mat = self.sigma_ch_exact_new()
        print("Sigma CH EXACT GPP")
        print(np.around((sigma_ch_exact_mat) * Sigma.sigma_factor, 6))
        print()
        # print(sigma_ch_exact_mat)
        # print()

        sigma_x_mat = self.sigma_x()
        print("Sigma X GPP")
        print(np.around((sigma_x_mat) * Sigma.sigma_factor, 6))
        print()
        # print(sigma_x_mat)
        # print()

        # Sig without Remainder
        sigma_mat = (
            sigma_x_mat
            + sigma_sx_gpp_mat
            + sigma_ch_gpp_mat
            + 0.5 * (sigma_ch_exact_mat - sigma_ch_static_mat)
        )
        sigma_mat = np.around((sigma_mat) * Sigma.sigma_factor, 6)
        print("Sig GPP:")
        print(sigma_mat.T)
        print("Eqp0")
        # print(sigma_mat.shape)
        # print(emf.shape)
        # print(vxc.shape)
        Eqp0 = sigma_mat + emf - vxc
        # Static Remainder
        # Eqp0 += Eqp0 + 0.5*(sigma_ch_exact_mat-sigma_ch_static_mat)[self.slice_l_k]* Sigma.sigma_factor
        print(Eqp0.T)

        # Calculate Eqp1
        # ==============
        # Assuming:
        # - finite_difference_form = 2:Forward difference
        # - finite_difference_spacing = 1.0
        # Note that sigma.inp docs are incorrect;
        # they mention Ecor, whereas, sigma_hp.log shows Eo, which is Emf for us

        dE[:] = 1.0
        dE /= emf_factor

        sigma_ch_gpp_mat_2 = self.sigma_ch_gpp(dE)
        print("Sigma CH GPP dE")
        print(np.around((sigma_ch_gpp_mat_2) * Sigma.sigma_factor, 6))
        print()
        # print(sigma_ch_gpp_mat_2)
        # print()
        # print(sigma_ch_gpp_mat_2 * Sigma.sigma_factor) #[self.slice_l_k]

        sigma_sx_gpp_mat_2 = self.sigma_sx_gpp(dE)
        print("Sigma SX GPP dE")
        print(np.around((sigma_sx_gpp_mat_2) * Sigma.sigma_factor, 6))
        print()
        # print(sigma_sx_gpp_mat_2)
        # print()
        # print(sigma_sx_gpp_mat_2 * Sigma.sigma_factor) #[self.slice_l_k]

        dSigdE = (
            (
                (sigma_sx_gpp_mat_2 + sigma_ch_gpp_mat_2)
                - (sigma_sx_gpp_mat + sigma_ch_gpp_mat)
            )
            / (dE * emf_factor)
            * Sigma.sigma_factor
        )
        slope = dSigdE / (1 - dSigdE)
        slope = slope  # [self.slice_l_k]
        Z = 1 / (1 - dSigdE)
        Z = Z  # [self.slice_l_k]
        print("Z:")
        print(Z.T)

        # Sig with Remainder
        sigma_mat_2 = (
            sigma_sx_gpp_mat_2 + sigma_ch_gpp_mat_2 + sigma_x_mat
        )  # [self.slice_l_k]
        sigma_mat_2 = np.around((sigma_mat_2) * Sigma.sigma_factor, 6)

        print("Sig_2:")
        print(sigma_mat_2.T)

        print("Eqp1")
        # Eqp1 = sigma_mat_2+emf-vxc
        # Eqp0_static_corrected = Eqp0 + 0.5*()
        Eqp1 = Eqp0 + slope * (Eqp0 - emf)
        print(np.real(Eqp1).T)

        print_x = np.real(
            np.around((sigma_x_mat) * Sigma.sigma_factor, 6)
        )  # [self.slice_l_k]
        print_sx = np.real(
            np.around((sigma_sx_gpp_mat) * Sigma.sigma_factor, 6)
        )  # [self.slice_l_k]
        print_ch = np.real(
            np.around((sigma_ch_gpp_mat) * Sigma.sigma_factor, 6)
        )  # [self.slice_l_k]
        print_exact_ch = np.real(
            np.around((sigma_ch_exact_mat) * Sigma.sigma_factor, 6)
        )  # [self.slice_l_k]

        for k in range(3):
            print(
                "   n         Emf          Eo           X        SX-X          CH         Sig         Vxc        Eqp0        Eqp1       Znk"
            )
            for n in range(self.sigmainp.band_index_min-1, self.sigmainp.band_index_max-self.sigmainp.band_index_min+1):
                print(
                    "{:>4}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}".format(
                        n,
                        emf[k, n],
                        emf[k, n],
                        print_x[k, n],
                        print_sx[k, n],
                        print_ch[k, n],
                        np.real(sigma_mat[k, n]),
                        np.real(vxc[k, n]),
                        np.real(Eqp0[k, n]),
                        np.real(Eqp1[k, n]),
                        np.real(Z[k, n]),
                    )
                )


if __name__ == "__main__":
    
    # Load WFN data

    from quantum_masala.gw.io_bgw import inp
    from quantum_masala.gw.io_bgw.epsmat_read_write import read_mats

    # sigmainp = inp.read_Sigma_inp()
    sigmainp = inp.read_sigma_inp(filename=dirname + "sigma.inp")

    # Sigma.inp data
    print(sigmainp.__doc__)
    print(sigmainp.options)
    print("sigmainp.kpts:", sigmainp.kpts)
    print()

    epsinp = inp.read_epsilon_inp(filename=dirname + "epsilon.inp")

    # Epsilon.inp data
    print(epsinp.__doc__)
    print(epsinp.options)
    print()

    # wfn2py
    from quantum_masala.gw.io_bgw.wfn2py import wfn2py

    wfndata = wfn2py(old_dirname + "WFN.h5")
    print(wfndata.__doc__)

    wfnqdata = wfn2py(old_dirname + "WFNq.h5")
    print(wfnqdata.__doc__)

    # To read QTMGW's epsmats:
    epsmats_dirname = "./test/epsilon/"
    l_epsmats_actual = []
    print(f"Reading epsilon matrices from directory: {epsmats_dirname}")
    l_epsmats_actual += read_mats(epsmats_dirname + "eps0mat_qtm.h5")
    l_epsmats_actual += read_mats(epsmats_dirname + "epsmat_qtm.h5")

    l_epsmats_bgw = []
    print(f"Reading epsilon matrices from directory: {epsmats_dirname}")
    l_epsmats_bgw += read_mats(dirname + "eps0mat.h5")
    l_epsmats_bgw += read_mats(dirname + "epsmat.h5")
    l_epsmats_bgw = [epsmat[0, 0] for epsmat in l_epsmats_bgw]
    l_epsmats_bgw = [
        epsmat[: min(epsmat.shape), : min(epsmat.shape)] for epsmat in l_epsmats_bgw
    ]

    # print(f"len l_epsmats {len(l_epsmats_actual)}")
    # print(f"shape l_epsmats[0] {l_epsmats_actual[0].shape}")
    # print(f"shape l_epsmats[1] {l_epsmats_actual[1].shape}")

    if global_test_epsinv:
        l_epsmats = l_epsmats_bgw
    else:
        l_epsmats = l_epsmats_actual

    def reorder_2d_matrix_sorted_gvecs(a, indices):
        """Given a 2-D matrix and listof indices, reorder rows and columns in order of indices"""
        tiled_indices = np.tile(indices, (len(indices), 1))
        return np.take_along_axis(
            np.take_along_axis(a, tiled_indices, 1), tiled_indices.T, 0
        )

    epsmats = []

    # %%
    from quantum_masala.gw.io_bgw.sigma_hp_reader import read_sigma_hp

    # sigma_dict = read_sigma_hp(dirname+'sigma_hp.log')
    sigma_dict = read_sigma_hp(old_dirname + "sigma_hp.log")

    # QPoints
    from quantum_masala.gw.core import QPoints

    # qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *epsinp.qpts)
    # print("qpts.cryst",qpts.cryst)

    # Sigma = Sigma(
    #     wfndata.crystal,
    #     wfndata.grho,
    #     wfndata.kpts,
    #     wfnqdata.kpts,
    #     wfndata.l_wfn,
    #     wfnqdata.l_wfn,
    #     wfndata.l_gwfn,
    #     wfnqdata.l_gwfn,
    #     qpts,
    #     sigmainp,
    # )

    sigma = Sigma.from_data(
        wfndata=wfndata,
        wfnqdata=wfnqdata,
        sigmainp=sigmainp,
        epsinp=epsinp,
        l_epsmats=l_epsmats,
    )

    # Run

    if int(sys.argv[1]) == 8:  # Sigma GPP
        print("Sigma GPP")
        sigma.calculate_gpp()

    if int(sys.argv[1]) == 9:  # Sigma Static COHSEX
        print("Sigma Static COHSEX")
        sigma.calculate_static_cohsex()

    if int(sys.argv[1]) == 0:  # Sigma SX STATIC COHSEX
        print("Sigma SX STATIC COHSEX")
        sigma_sx_static_mat = sigma.sigma_sx_static()
        print("Sigma SX STATIC COHSEX")
        print()
        print(sigma_sx_static_mat)
        print()
        print(np.around((sigma_sx_static_mat) * Sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 1:  # Sigma SX GPP
        print("Sigma SX GPP")
        sigma_sx_gpp_mat = sigma.sigma_sx_gpp()
        print("Sigma SX GPP")
        print()
        print(sigma_sx_gpp_mat)
        print()
        print(np.around((sigma_sx_gpp_mat) * Sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 2:  # Sigma CH GPP
        print("Sigma CH GPP")
        sigma_ch_gpp_mat = sigma.sigma_ch_gpp()
        print("Sigma CH GPP")
        print()
        print(sigma_ch_gpp_mat)
        print()
        print(np.around((sigma_ch_gpp_mat) * Sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 3:  # Sigma CH STATIC COHSEX
        print("Sigma CH STATIC COHSEX")
        sigma_ch_static_mat = sigma.sigma_ch_static()
        print("Sigma CH STATIC COHSEX")
        print()
        print(sigma_ch_static_mat)
        print()
        print(np.around((sigma_ch_static_mat) * Sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 4:  # Sigma CH EXACT GPP
        print("Sigma CH EXACT GPP")
        sigma_ch_exact_mat = sigma.sigma_ch_exact_new()
        print("Sigma CH EXACT GPP")
        print()
        print(sigma_ch_exact_mat)
        print()
        print(np.around((sigma_ch_exact_mat) * Sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 5:  # Sigma X
        print("Sigma X")
        sigma_x_mat = sigma.sigma_x()
        print("Sigma X GPP")
        print()
        print(sigma_x_mat)
        print()
        print(np.around((sigma_x_mat) * Sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 6:  # Sigma CH GPP dE
        # number_bands = sigma.sigmainp.number_bands
        # emf_factor = 5.790268/0.4255769
        # emf = np.array([wfn.evl[0] for wfn in sigma.l_wfn]) * emf_factor
        # dE = np.zeros_like(emf[:,:number_bands])
        # dE[self.slice_l_k] = 1.0
        # dE /= emf_factor

        dE = 1.0 / Sigma.ryd

        print("Sigma CH GPP dE")
        sigma_ch_gpp_mat_2 = sigma.sigma_ch_gpp(dE)
        print("Sigma CH GPP dE")
        print()
        print(sigma_ch_gpp_mat_2)
        print()
        print(np.around((sigma_ch_gpp_mat_2) * Sigma.sigma_factor, 6))
        # print(sigma_ch_gpp_mat_2 * Sigma.sigma_factor) #[self.slice_l_k]

    if int(sys.argv[1]) == 7:  # Sigma SX GPP dE
        # number_bands = sigma.sigmainp.number_bands
        # emf_factor = 5.790268/0.4255769
        # emf = np.array([wfn.evl[0] for wfn in sigma.l_wfn]) * emf_factor
        # dE = np.zeros_like(emf[:,:number_bands])
        # dE[self.slice_l_k] = 1.0
        # dE /= emf_factor

        dE = 1.0 / Sigma.ryd

        print("Sigma SX GPP dE")
        sigma_sx_gpp_mat_2 = sigma.sigma_sx_gpp(dE)
        print("Sigma SX GPP dE")
        print()
        print(sigma_sx_gpp_mat_2)
        print()
        print(np.around((sigma_sx_gpp_mat_2) * Sigma.sigma_factor, 6))

    print(
        "Sigma script finished running : ",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
