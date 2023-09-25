"""
Sigma
=====
"""

from collections import namedtuple
import datetime
import sys
from copy import deepcopy
from functools import lru_cache
from typing import List, NamedTuple

import numpy as np
from qtm.gw.epsilon import Epsilon
from tqdm import trange

from qtm.constants import ELECTRONVOLT_RYD, RYDBERG_HART, ELECTRONVOLT_HART
from qtm.crystal import Crystal
from qtm.dft.kswfn import KSWfn
from qtm.fft.backend.utils import get_fft_driver
from qtm.gspace import GkSpace, GSpace
from qtm.gspace.base import cryst2idxgrid
from qtm.gw.core import QPoints, reorder_2d_matrix_sorted_gvecs, sort_cryst_like_BGW
from qtm.gw.io_bgw.wfn2py import WfnData
from qtm.gw.io_h5.h5_utils import *
from qtm.gw.vcoul import Vcoul
from qtm.klist import KList
from qtm.mpi.comm import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py import MPI
    from mpi4py.MPI import COMM_WORLD

# -------- Constant Parameters -------------------
NDIGITS = 5
TOLERANCE = 1e-5

# -------- Debugging params ----------------------
global_test_vcoul = False  # True may not work for now, due to broken sorted-gpts


class Sigma:
    """Sigma Matrix Class"""

    ryd = 1 / ELECTRONVOLT_RYD
    limitone = 1 / TOLERANCE / 4  # limitone=1D0/(TOL_Small*4D0)
    limittwo = 0.25  # limittwo=sig%gamma**2
    fixwings = True
    autosave = False

    # FIXME: Why is l_gsp_wfn being passed separately, in addition to l_wfn? The KSWfn objects contain a GkSpace object already.

    def __init__(
        self,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        kptsq: KList,
        l_wfn: List[KSWfn],
        l_wfnq: List[KSWfn],
        l_gsp_wfn: List[GkSpace],
        l_gsp_wfnq: List[GkSpace],
        qpts: QPoints,
        sigmainp: NamedTuple,
        epsinp: NamedTuple,
        l_epsmats: List[np.ndarray],
        rho: NamedTuple,
        vxc: NamedTuple,
        outdir: str = None,
        parallel: bool = True,
    ):
        """Initialize Sigma

        - Receive GSpace, ElectronWfn etc. objects constructed from ``wfn.h5`` and ``wfnq.h5``
        - Load SigmaInp object constructed from ``sigma.inp``

        Parameters
        ----------
        crystal : Crystal
        gspace : GSpace
        kpts : KList
        kptsq : KList
        l_wfn : List[KSWfn]
            List of (unshifted) wavefunctions
        l_wfnq : List[KSWfn]
            List of shifted wavefunctions
        l_gsp_wfn : List[GkSpace]
            _description_
        l_gsp_wfnq : List[GkSpace]
            _description_
        qpts : QPoints
            _description_
        sigmainp : NamedTuple
            _description_
        epsinp : NamedTuple
            _description_
        l_epsmats : List[np.ndarray]
            _description_
        rho : NamedTuple
            _description_
        vxc : NamedTuple
            _description_
        outdir : str, optional
            _description_, by default None
        parallel : bool, optional
            _description_, by default True
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
        self.rho = rho
        self.vxc = vxc
        self.outdir = outdir

        self.in_parallel = False
        self.comm = None
        self.comm_size = None
        if parallel:
            if MPI4PY_INSTALLED:
                self.comm = MPI.COMM_WORLD
                self.comm_size = self.comm.Get_size()
                if self.comm.Get_size() > 1:
                    self.in_parallel = True

        if self.outdir is None:
            self.outdir = "./"

        # Read rho and create hash map for rhogrid
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
        self.sigma_factor = Sigma.ryd / (crystal.reallat.cellvol * qpts.numq)

        # K-Points (typically from SigmaInp)
        self.l_k_indices = []
        for kpt in self.sigmainp.kpts[:, :3]:
            self.l_k_indices.append(
                np.where(np.all(kpt == self.kpts.cryst, axis=1))[0][0]
            )

        self.l_k_indices = np.array(self.l_k_indices)
        self.n_kpts = len(self.l_k_indices)
        self.slice_l_k = np.s_[self.l_k_indices]

        self.l_gq: List[GkSpace] = []
        for i_q in range(qpts.numq):
            self.l_gq.append(
                GkSpace(
                    gwfn=self.gspace,
                    k_cryst=self.qpts.cryst[i_q],
                    ecutwfn=self.epsinp.epsilon_cutoff * RYDBERG_HART,
                )
            )

        self.sorted_gpts = None

        self.vcoul = Vcoul(
            gspace=self.gspace,
            qpts=self.qpts,
            bare_coulomb_cutoff=sigmainp.bare_coulomb_cutoff,
            parallel=self.in_parallel,
        )

        self.print_condition = (not self.in_parallel) or (
            self.in_parallel and self.comm.Get_rank() == 0
        )
        self.print_condition_comm = self.print_condition and (self.comm != None)
        if self.print_condition:
            print("vcoul:", self.vcoul)

        self.vcoul.calculate_vcoul(averaging_func=self.vcoul.v_minibz_montecarlo_hybrid)

        # "Epsilon Inverse minus Identity" matrices
        self.l_epsinv_I = []

        for i_q in range(self.qpts.numq):
            # print("i_q", i_q)

            epsinv = self.l_epsmats[i_q]

            # Fix wings (look at Table 2 in BerkeleyGW Paper (2012))
            if self.fixwings:
                epsinv = self.vcoul.calculate_fixedeps(epsinv, i_q, random_sample=False)
            epsinv_I = epsinv - np.eye(len(epsinv))

            sort_order = sort_cryst_like_BGW(
                self.l_gq[i_q].gk_cryst, self.l_gq[i_q].gk_norm2
            )

            self.l_epsinv_I.append(
                reorder_2d_matrix_sorted_gvecs(epsinv_I, np.argsort(sort_order))
            )

        return

    @classmethod
    def from_qtm_scf(
        cls,
        crystal: Crystal,
        gspace: GSpace,
        kpts: KList,
        kptsq: KList,
        l_wfn_kgrp: List[List[KSWfn]],
        l_wfn_kgrp_q: List[List[KSWfn]],
        sigmainp: NamedTuple,
        epsinp: NamedTuple,
        epsilon:Epsilon,
        rho: NamedTuple,
        vxc: NamedTuple,
        outdir: str = None,
        parallel: bool = True,       
    ):
        rho_temp = deepcopy(rho)
        rho_temp._data *= crystal.numel / (sum(rho_temp.data_g0))

        rho_nt = namedtuple("RHO", ["rho", "gvecs"])(rho_temp.data[0], rho_temp.gspc.g_cryst.T)
        vxc_nt = namedtuple("VXC", ["vxc"])(vxc/ELECTRONVOLT_HART)


        sigma = Sigma(
            crystal = crystal,
            gspace = gspace,
            kpts = kpts,
            kptsq=kptsq,
            l_wfn = [wfn[0] for wfn in l_wfn_kgrp],
            l_wfnq = [wfn[0] for wfn in l_wfn_kgrp_q],
            l_gsp_wfn = [wfn[0].gkspc for wfn in l_wfn_kgrp],
            l_gsp_wfnq = [wfn[0].gkspc for wfn in l_wfn_kgrp_q],
            qpts = QPoints.from_cryst(kpts.recilat, epsinp.is_q0, *epsinp.qpts),
            sigmainp=sigmainp,
            epsinp = epsinp,
            l_epsmats=epsilon.l_epsinv,
            rho=rho_nt,
            vxc=vxc_nt,
            outdir=outdir,
            parallel=parallel
        )
        return sigma

    @classmethod
    def from_data(
        cls,
        wfndata: WfnData,
        wfnqdata: WfnData,
        sigmainp: NamedTuple,
        epsinp: NamedTuple,
        l_epsmats: List[np.ndarray],
        rho: NamedTuple,
        vxc: NamedTuple,
        outdir: str = None,
    ):
        # FIXME: Needs a more robust logic, i.e. we should read qpts from epsmat.h5 etc., as is done in BerkeleyGW/Sigma
        #        That is, we shouldn't require epsilon.inp file for Sigma calculation.
        #        Ideally, epsmat.h5 should contain all the data that we need,
        #        thus ensuring consistency of data.
        l_qpts = np.array(epsinp.qpts)
        l_qpts[0] *= 0
        qpts = QPoints.from_cryst(wfndata.kpts.recilat, None, *l_qpts)

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
            rho,
            vxc,
            outdir,
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
        if np.linalg.norm(qpt) < TOLERANCE or self.qpts.index_q0 == i_q:
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

    def map_g_to_g_minusq(self, g, g_at_minusq, q, minusq):
        """Return the indices in g_at_minusq that corresponds to respective -g vectors
        Consider making a hash-map to reduce search time.

        Returns
        -------
            Indices in g_minusq that corresponds to respective -g vectors.
        """
        # Initialize indices to -1's, which will indicate "not found" case
        # "Not found" case will never occur because we are using np.argmin to get an index
        # Still, we need to initialize with something,
        # and it's not a bad idea to initialize with -1
        indices = -1 * np.ones(len(g), dtype=int)

        for i, gvec in enumerate(g):
            target = -1 * (gvec + q)
            sub_norm = np.linalg.norm((g_at_minusq + minusq) - target[None, :], axis=1)
            i_g_minusq = np.argmin(sub_norm, axis=0)
            if sub_norm[i_g_minusq] > 1e-5:
                print("Problem in map_g_to_g_minusq", q, sub_norm[i_g_minusq])
                print("g_minusq[i_g_minusq]", g_at_minusq[i_g_minusq])
                print("minusq", minusq)
            indices[i] = i_g_minusq

        return indices

    def find_indices(self, l_targets, l_list):
        """Find indices of elements matching the targets in list.
        Returns list of indices corresponding to target vector. -1 if not found"""
        ret = []
        for target in l_targets:
            if target in l_list:
                ret.append(l_list.index(target))
            else:
                ret.append(-1)
        return ret

    def pprint_sigma_mat(self, mat):
        _mat = deepcopy(mat).real.T
        print("  n  ",end="")
        print(("    ik={:<5}" * _mat.shape[-1]).format(*self.l_k_indices+1))
        
        for i,row in enumerate(_mat.reshape(-1, _mat.shape[-1])):
            print(f"{i+self.sigmainp.band_index_min:>3} ",end="")
            print(("{:12.6f}" * _mat.shape[-1]).format(*np.around(row, 6)))

    # ==================================================================
    # Plane wave matrix elements calculation methods

    # @pw_logger.time("sigma:matrix_elements")
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

        # Load Sigma.inp data
        is_q0 = self.qpts.is_q0  # np.array(inp.qpts[:, 4], dtype=bool)
        l_q = self.qpts.cryst  # list of q-points in crystal coords

        # Occupation numbers

        occ_all_bands = []
        for i_k in range(self.kpts.numk):  # self.l_k_indices:
            occ_all_bands.append(self.l_wfn[i_k].occ)
        occ_all_bands = np.array(occ_all_bands)

        occ = occ_all_bands[:, 0 : self.sigmainp.number_bands]
        # ^ indices for reference: [index of kpoint, band index]

        l_i_v = np.where(occ == 1)  # list of indices of occupied   bands
        l_i_c = np.where(occ == 0)  # list of indices of unoccupied bands

        prod_grid_shape = np.prod(self.gspace.grid_shape)

        n_v_max = max(l_i_v[1]) + 1
        i_c_beg = min(l_i_c[1])

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
        @lru_cache()
        def get_evc_gk_g2r(i_k, i_b, checkq0=True):
            if is_qpt_0 and checkq0:
                wfn = self.l_wfnq[i_k]
                gkspc = self.l_gsp_wfnq[i_k]
            else:
                wfn = self.l_wfn[i_k]
                gkspc = self.l_gsp_wfn[i_k]

            arr_r = np.zeros(self.gspace.grid_shape, dtype=complex)
            gkspc._fft.g2r(wfn.evc_gk._data[i_b, :], arr_out=arr_r)
            return arr_r

        # MATRIX ELEMENTS CALCULATION -------------------

        # Handling `_all_bands` parameters
        if bra_all_bands:
            n_bra = self.sigmainp.number_bands
            i_b_bra_beg = 0  # Index where valence bands begin (always 0)
        else:
            n_bra = n_v_max
            i_b_bra_beg = 0

        if ket_all_bands:
            i_b_ket_beg = (
                self.sigmainp.band_index_min - 1
            )  # Index where conduction bands begin
            n_ket = self.sigmainp.band_index_max - self.sigmainp.band_index_min + 1
        else:
            i_b_ket_beg = i_c_beg
            n_ket = self.sigmainp.band_index_max - (i_b_ket_beg + 1) + 1

        # Init M
        if yielding:
            M = np.zeros((self.n_kpts, n_ket, self.l_gq[i_q].size_g), dtype=complex)
        else:
            M = np.zeros(
                (self.n_kpts, n_bra, n_ket, self.l_gq[i_q].size_g), dtype=complex
            )

        # Find pairs of ket and bra k-points
        # They are related as follows:
        # k_ket + qpt = k_bra
        pairs_i_k = []
        for i_k_ket in l_k_indices:
            k_bra_indices = np.where(
                (np.isclose(l_k, l_kplusq[i_k_ket], TOLERANCE)).all(axis=1)
            )[0]
            assert (
                len(k_bra_indices) > 0
            ), f"Could not find k-point in wavefunction k-points that matches k+q.\nHere is the list of k-points:{l_k}\nand here is the k+q vector:{l_kplusq[i_k_ket]}"
            pairs_i_k.append((i_k_ket, k_bra_indices[0]))

        if ret_E == True:
            E_ket = np.zeros((self.n_kpts, n_ket), dtype=complex)
            E_bra = np.zeros((self.n_kpts, n_bra), dtype=complex)
            for i_k_ket, i_k_bra in pairs_i_k:
                i_k_ket_in_mtxel_call = np.where(self.l_k_indices == i_k_ket)[0][0]
                for i_b_bra in range(n_bra):
                    E_bra[i_k_ket_in_mtxel_call, i_b_bra] = self.l_wfn[i_k_bra].evl[
                        i_b_bra + i_b_bra_beg
                    ]  * 2
                for i_b_ket in range(n_ket):
                    E_ket[i_k_ket_in_mtxel_call, i_b_ket] = self.l_wfn[i_k_ket].evl[
                        i_b_ket + i_b_ket_beg
                    ]  * 2

        # Matrix elements calculation

        phi_bra = {}
        phi_ket = {}

        for i_b_bra in range(n_bra):
            for i_k_ket, i_k_bra in pairs_i_k:
                i_k_ket_in_mtxel_call = np.where(l_k_indices == i_k_ket)[0][0]

                l_g_umklapp = self.l_gq[i_q].g_cryst - umklapp[i_k_ket][:, None]
                idxgrid = cryst2idxgrid(
                    shape=self.gspace.grid_shape, g_cryst=l_g_umklapp.astype(int)
                )

                grid_g_umklapp = tuple(
                    np.mod(
                        l_g_umklapp, np.array(self.gspace.grid_shape)[:, None]
                    ).astype(int)
                )

                umklapped_fft_driver = get_fft_driver()(
                    self.gspace.grid_shape,
                    idxgrid,
                    normalise_idft=False,
                )

                phi_bra[i_b_bra] = (
                    get_evc_gk_g2r(i_k_bra, i_b_bra + i_b_bra_beg, checkq0=True)
                    / prod_grid_shape
                )

                for i_b_ket in range(n_ket):
                    phi_ket[i_b_ket] = (
                        get_evc_gk_g2r(i_k_ket, i_b_ket + i_b_ket_beg, checkq0=False)
                        / prod_grid_shape
                    )

                    prod = np.multiply(np.conj(phi_ket[i_b_ket]), phi_bra[i_b_bra])
                    fft_prod = np.zeros(
                        umklapped_fft_driver.idxgrid.shape, dtype=complex
                    )
                    umklapped_fft_driver.r2g(prod, fft_prod)
                    if yielding:
                        M[i_k_ket_in_mtxel_call, i_b_ket] = (prod_grid_shape) * fft_prod
                    else:
                        M[i_k_ket_in_mtxel_call, i_b_bra, i_b_ket] = (
                            prod_grid_shape
                        ) * fft_prod
            if yielding:
                if ret_E == True:
                    yield M, E_bra[:, i_b_bra], E_ket
                else:
                    yield M

        if not yielding:
            if ret_E == True:
                yield M, E_bra, E_ket
            else:
                yield M

    # #@pw_logger.time('sigma:matrix_elements_sigma_exact')
    def matrix_elements_sigma_exact(self, l_g):
        """
        To Calculate the matrix elements for exact sigma_CH.

            M_{n,n'} (k,q,G) = < v, k | exp(i(G'-G).r) | c, k >

        Returns M with shape (n_valence, n_conduction, number of G-vectors within cutoff for q=0)

        TODO: Make this a generator like `matrix_elements`
        """
        # k points data -------------------------------------------------
        n_kpts = self.qpts.numq

        # list of k-points in crystal coords
        l_k = self.kpts.cryst  # kpoints.rk

        # Load Sigma.inp data --------------------------------------------
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max - band_index_min + 1

        # list of q-points in crystal coords
        epsilon_cutoff = self.epsinp.epsilon_cutoff

        l_i_sv_tuple = np.meshgrid(list(range(number_bands_outer)), list(range(n_kpts)))
        l_i_sv = np.array([l_i_sv_tuple[1].flatten(), l_i_sv_tuple[0].flatten()])
        l_i_v_m = l_i_sv
        n_v_m = number_bands_outer

        l_i_sc_tuple = np.meshgrid(list(range(number_bands_outer)), list(range(n_kpts)))
        l_i_sc = np.array([l_i_sc_tuple[1].flatten(), l_i_sc_tuple[0].flatten()])
        l_i_c_m = l_i_sc
        n_c_m = number_bands_outer

        M = np.zeros((self.n_kpts, n_v_m, len(l_g[0])), dtype=complex)

        @lru_cache()
        def get_evc_gk_r2g(i_k, i_b):
            wfn_v = self.l_wfn[i_k]
            gkspc_v = self.l_gsp_wfn[i_k]

            phi_v = np.zeros(self.gspace.grid_shape, dtype=complex)
            gkspc_v._fft.g2r(wfn_v.evc_gk._data[i_b, :], arr_out=phi_v)
            return phi_v

        # MATRIX CALCULATION -------------------

        prev_i_k_ket = None  # To avoid recalculation for the same value of k
        for i_c in range(n_kpts * n_c_m):
            i_k_ket = l_i_c_m[0][i_c]  # unoccupied k indices, repeated
            i_b_ket = l_i_c_m[1][i_c]  # unoccupied band indices, repeated

            if i_k_ket not in self.l_k_indices:
                continue

            i_k_c_insigma = np.where(self.l_k_indices == i_k_ket)[0][0]
            # first [0] index for tuple, the second [0] to get the first (and the only) match

            wfn_ket = self.l_wfn[i_k_ket]
            gkspc_ket = self.l_gsp_wfn[i_k_ket]
            phi_ket = np.zeros(self.gspace.grid_shape, dtype=complex)
            gkspc_ket._fft.g2r(wfn_ket.evc_gk._data[i_b_ket, :], arr_out=phi_ket)

            if prev_i_k_ket != i_k_ket:  # 'if' to avoid re-calculation
                prev_i_k_ket = i_k_ket

                idxgrid = cryst2idxgrid(
                    shape=self.gspace.grid_shape, g_cryst=l_g.astype(int)
                )

                umklapped_fft_driver = get_fft_driver()(
                    self.gspace.grid_shape,
                    idxgrid,
                    normalise_idft=False,
                    skip_check_g_idxgrid_len=True,
                    # NOTE: normalise_idft=False will be the default for all gw code
                )

            # k_c plus q for the cuurent i_k_c
            kcplusq = l_k[i_k_ket]

            # obtain a list of indices of valence kpoints that match  k_c + q
            # In this case, q=0, which is at 0 index in the valence (bra) list
            # [ k-index , component index: 0,1,2 ]
            l_k_bra = l_k[l_i_v_m[0][:], :]

            l_i_match = np.nonzero(
                np.all(np.isclose(l_k_bra, kcplusq[None, :], atol=1e-5), axis=1)
            )[0]

            # for k_v == k_c + q:
            for i_v in l_i_match:
                i_k_bra = l_i_v_m[0][i_v]  # occupied k indices, repeated
                i_b_bra = l_i_v_m[1][i_v]  # occupied band indices
                if i_b_bra != i_b_ket:
                    continue

                phi_bra = get_evc_gk_r2g(i_k_bra, i_b_bra)
                prod = np.multiply(np.conj(phi_ket), phi_bra)
                fft_prod = np.zeros(umklapped_fft_driver.idxgrid.shape, dtype=complex)
                umklapped_fft_driver.r2g(prod, fft_prod)

                M[i_k_c_insigma, i_b_bra] = fft_prod

        for i, g in enumerate(l_g.T):
            normsq_g = self.gspace.recilat.norm(g) ** 2
            if normsq_g > epsilon_cutoff:
                M[:, :, i] *= 0

        return M

    # ==================================================================
    # Static Sigma methods

    # @pw_logger.time("sigma:sigma_x")
    def sigma_x(self, yielding=True, parallel=True):
        """
        Fock exchange energy term
        =========================
        Returns Sigma_x[i_k, i_band] for diag = True

        - Sum _(n" over occupied, q, G, G')   M^*_n"n'(k,-q,-G)  M_n"n'(k,-q,-G')  delta_GG'  v(q+G')
        - Sum _(n" over occupied, q, G=G')    M^*_n"n'(k,-q,-G)  M_n"n'(k,-q,-G)  v(q+G)
        """

        # Init empty sigma matrix
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max - band_index_min + 1
        sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)

        # n   : k-points
        # i   : valence    states
        # j,l : conduction states
        # k   : G
        # m   : G'
        # in sigma_x the delta_GG' means k=m

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0

        gpts = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]

        # Embarassingly parallel, just use microprocessing library
        def compute_sigma_q(i_q):
            # Get qpt value
            qvec = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            minus_qvec = l_q[i_minusq]

            # Calculate map from G to -G : map_g_to_minusg
            g_minusq = gpts[i_minusq].T
            g = gpts[i_q].T
            g_to_mg = self.map_g_to_g_minusq(g, g_minusq, qvec, minus_qvec)

            vqg = self.vcoul.vcoul[i_q]

            sigma_q = np.zeros_like(sigma)

            # yielding: To reduce memory consumption
            if yielding:
                einstr = "njm,njm,m->nj"
                for M in self.matrix_elements(
                    i_minusq, ket_all_bands=True, yielding=yielding
                ):
                    M = M[..., g_to_mg]
                    sigma_q += np.einsum(einstr, np.conj(M), M, vqg, optimize=True)

            else:
                einstr = "nijm,nijm,m->nj"
                M = next(
                    self.matrix_elements(
                        i_minusq, ket_all_bands=True, yielding=yielding
                    )
                )[..., g_to_mg]
                sigma_q = np.einsum(einstr, np.conj(M), M, vqg, optimize=True)

            return sigma_q

        if not (self.in_parallel and parallel):
            proc_q_indices = range(len(l_q))
            for i_q in trange(len(l_q), desc="Sigma_X"):
                sigma_q = compute_sigma_q(i_q)
                sigma -= sigma_q
        else:
            proc_rank = self.comm.Get_rank()
            q_indices = np.arange(self.qpts.numq)
            proc_q_indices = np.array_split(q_indices, self.comm_size)[proc_rank]

            for i_q in proc_q_indices:
                print(
                    f"Rank: {self.comm.Get_rank()}, i_q: {i_q}, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                sigma_q = compute_sigma_q(i_q)
                sigma -= sigma_q
            gathered_sigma = self.comm.allgather(sigma)

            sigma = sum(gathered_sigma)

        sigma *= self.sigma_factor

        if Sigma.autosave:
            np.save(
                self.outdir
                + f"sigma_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sigma_x_{proc_q_indices[0]}",
                sigma,
            )

        return sigma

    # @pw_logger.time("sigma:sigma_sx_static")
    def sigma_sx_static(self, yielding=True):
        """
        Static Screened Exchange
        ========================
        """

        # Setting einstein summation string for M* M epsinv v

        # Init empty sigma matrix
        # number_bands_inner = self.sigmainp.number_bands
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max - band_index_min + 1
        # sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)

        sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
        einstr = "nijk,nijm,mk,m->nj"

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0
        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]
        sort_order = sort_cryst_like_BGW(self.l_gq[0].gk_cryst, self.l_gq[0].gk_norm2)
        print(sort_order.shape)

        if not self.in_parallel:
            iterable = trange(len(l_q), desc="Sigma_SX_Static")
            proc_q_indices = range(len(l_q))
        else:
            proc_rank = self.comm.Get_rank()
            q_indices = np.arange(self.qpts.numq)
            proc_q_indices = np.array_split(q_indices, self.comm_size)[proc_rank]
            iterable = proc_q_indices

        for i_q in iterable:
            # Get qpt value
            qpt = l_q[i_q]

            # Find index of -q
            i_minusq = self.index_minusq(i_q)
            mqpt = l_q[i_minusq]

            # Calculate map from G to -G : map_g_to_minusg
            g_minusq = g_cryst[i_minusq].T
            g = g_cryst[i_q].T
            g_to_mg = self.map_g_to_g_minusq(g, g_minusq, qpt, mqpt)

            epsinv_I = self.l_epsinv_I[i_q]

            if i_q == self.qpts.index_q0:
                key = self.l_gsp_wfnq[i_q].gk_norm2
            else:
                key = self.l_gsp_wfn[i_q].gk_norm2

            indices_gspace_sorted = sort_cryst_like_BGW(
                cryst=self.l_gsp_wfn[i_q].gk_cryst, key_array=key
            )

            vqg = self.vcoul.vcoul[i_q]

            # Calculate matrix elements
            # yielding: To reduce memory consumption
            if yielding:
                einstr = "njk,njm,km,k->nj"
                for M in self.matrix_elements(
                    i_minusq, ket_all_bands=True, yielding=yielding
                ):
                    M = M[..., g_to_mg]
                    sigma += np.einsum(
                        einstr,
                        np.conj(M),
                        M,
                        epsinv_I,
                        vqg,
                        optimize=True,
                    )

            else:
                einstr = "nijk,nijm,km,k->nj"
                M = next(
                    self.matrix_elements(
                        i_minusq, ket_all_bands=True, yielding=yielding
                    )
                )
                M = M[..., g_to_mg]
                sigma += np.einsum(
                    einstr,
                    np.conj(M),
                    M,
                    epsinv_I,
                    vqg,
                    optimize=True,
                )

            # sigma += sigma_q
            # i   : valence    states
            # j,l : conduction states
            # k   : G
            # m   : G'

        if self.in_parallel:
            sigma = sum(self.comm.allgather(sigma))

        sigma *= -1 * self.sigma_factor

        if Sigma.autosave:
            np.save(
                self.outdir
                + f"sigma_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sigma_sxstatic_{proc_q_indices[0]}",
                sigma,
            )

        return sigma

    # @pw_logger.time("sigma:sigma_ch_static")
    def sigma_ch_static(self, yielding=True):
        """
        Static Coulomb Hole (partial sum)
        =================================
        """

        # Setting einstein summation string for M* M epsinv v
        # Init empty sigma matrix
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max - band_index_min + 1

        # if diag:
        sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords
        l_q[self.qpts.index_q0] *= 0

        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]

        # --------------------------------------------------

        if not self.in_parallel:
            iterable = trange(len(l_q), desc="Sigma_CH_Static_Partial")
            proc_q_indices = range(len(l_q))
        else:
            proc_rank = self.comm.Get_rank()
            q_indices = np.arange(self.qpts.numq)
            proc_q_indices = np.array_split(q_indices, self.comm_size)[proc_rank]
            iterable = proc_q_indices

        for i_q in iterable:
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

            # Get ususal Coulomb potential: vqg = vcoulomb(g)
            vqg = self.vcoul.vcoul[i_q]

            # yielding: To reduce memory consumption
            if yielding:
                einstr = "njk,njm,km,k->nj"
                for M in self.matrix_elements(
                    i_minusq, ket_all_bands=True, bra_all_bands=True, yielding=yielding
                ):
                    M_ = M[..., g_to_mg]
                    sigma += np.einsum(
                        einstr,
                        np.conj(M_),
                        M_,
                        epsinv_I,
                        vqg,
                        optimize=True,
                    )

            else:
                einstr = "nijk,nijm,km,k->nj"
                # i   : valence    states
                # j,l : conduction states
                # k   : G
                # m   : G'
                M = next(
                    self.matrix_elements(
                        i_minusq,
                        ket_all_bands=True,
                        bra_all_bands=True,
                        yielding=yielding,
                    )
                )[..., g_to_mg]
                sigma += np.einsum(
                    einstr,
                    np.conj(M),
                    M,
                    epsinv_I,
                    vqg,
                    optimize=True,
                )

        if self.in_parallel:
            sigma = sum(self.comm.allgather(sigma))

        sigma *= 0.5 * self.sigma_factor

        return sigma

    # @pw_logger.time("sigma:sigma_ch_static_exact")
    def sigma_ch_static_exact(self):
        """
        Static Coulomb Hole (Exact)
        ========================
        0.5 * \Sum_{q,G,G'} = M_{n,n'}(k, q=0, G'-G) * [\eps^{-1}_{G,G'}(q;0) - \delta_{G,G'}] * v(q+G')

        1e-3 disagreement: Doubt goes to limits on G'-G: is it all G and G' within cutoff or G-G' within cutoff
        """

        # Init empty sigma matrix
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max - band_index_min + 1
        sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords

        # Assume received epsinv and matrix elements ordered by l_q
        if not self.in_parallel:
            iterable = trange(len(l_q), desc="Sigma_CH_Static_Exact")
            proc_q_indices = range(len(l_q))
        else:
            proc_rank = self.comm.Get_rank()
            q_indices = np.arange(self.qpts.numq)
            proc_q_indices = np.array_split(q_indices, self.comm_size)[proc_rank]
            iterable = proc_q_indices

        for i_q in iterable:
            g = self.l_gq[i_q].g_cryst.T

            # G'-G = gpmg = Gprime minus G
            gpmg = np.array([[gpvec - gvec for gpvec in g] for gvec in g])

            # Get epsinv-I
            epsinv_I = self.l_epsinv_I[i_q]

            # Get ususal Coulomb potential: vqg = vcoulomb(g)
            vqg = self.vcoul.vcoul[i_q]

            # Calculate matrix elements
            M = self.matrix_elements_sigma_exact(np.vstack(gpmg).T)

            # Reshape M from all possible (G'-G) to (G',G).
            # We had used vstack with gpmg to create a list of g-vectors
            # to be sent to Matrix_elemts_augmented function.
            # Therefore, we need to unpack the result.
            # Ensure that the result does the same cutoff as BGW with G'-G.
            # !For now, assuming nothing special is done to cutoff |G'-G|.

            len_g = len(g)
            M = np.array([M[..., len_g * i : len_g * (i + 1)] for i in range(len_g)])

            einstr = "kijm,mk,m->ij"
            sigma += np.einsum(einstr, M, epsinv_I, vqg, optimize=True) / np.prod(
                self.gspace.grid_shape
            )
            # i   : kpoint index
            # j   : band index
            # k   : G
            # m   : G'

        if self.in_parallel:
            sigma = sum(self.comm.allgather(sigma))

        sigma *= 0.5 * self.sigma_factor

        return sigma

    # ==================================================================
    # GPP Sigma methods

    # @pw_logger.time("sigma:sigma_gpp_omegas")
    def sigma_gpp_omegas(self, qpt, g, vqg, epsinv_I):
        r"""GPP Omega Calculator

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
            abs(1_GG' - epsinv_GG'(q;0)), or
            abs(λ_GG'(q)), or
            abs(cos(φ_GG'(q)))
            are less than a given tolerance, since these terms have
            a vanishing contribution to the matrix elements of the self-energy.
            This avoids ill-conditioned limits due to some of the intermediate
            quantities here being undeﬁned.
        """
        # i_g0 : index of (0,0,0) vector in the gvecs list of rho
        # required for calculing rho(0)

        # RHO data
        rho = self.rho

        i_g0 = i_gmgpvec = self.rho_dict[self.rhohash([0, 0, 0])]
        if i_g0 < 0:  # The difference does not exist in rho gvecs list
            raise ValueError("G=0 vector not found in G-vectors list of RHO.")
        nelec = rho.rho[i_g0]

        # define ω_p^2
        wp2 = self.ryd**2 * 16 * np.pi * nelec / self.crystal.reallat.cellvol

        # init matrices
        len_g = len(g)
        omega2 = np.zeros((len_g, len_g), dtype=complex)
        wtilde2 = np.zeros((len_g, len_g), dtype=complex)
        wpmtx = np.zeros((len_g, len_g), dtype=complex)

        for i_gp, gpvec in enumerate(g):
            for i_g, gvec in enumerate(g):
                i_gmgpvec = self.rho_dict[self.rhohash(gvec - gpvec)]
                if i_gmgpvec < 0:  # The difference does not exist in rho gvecs list
                    continue  # In accordance with BGW. See #1

                # Correct rho
                rho_gmgpvec = rho.rho[i_gmgpvec].copy()
                if np.imag(rho_gmgpvec) > TOLERANCE:
                    rho_gmgpvec *= 0

                # Skip small I-epsinv
                if abs(epsinv_I[i_g, i_gp]) < TOLERANCE:
                    continue  # See #2

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

                wtilde2_temp = omega2[i_g, i_gp] / (-epsinv_I[i_g, i_gp])

                lambda_abs = np.absolute(wtilde2_temp)

                # Skip small lambda
                if abs(lambda_abs) < TOLERANCE:
                    continue  # In accordance with BGW, See #4

                phi = np.arctan2(np.imag(wtilde2_temp), np.real(wtilde2_temp))

                # Skip small cos(phi)
                if abs(np.cos(phi)) < TOLERANCE:
                    continue  # In accordance with BGW, See #4

                wtilde2[i_g, i_gp] = lambda_abs / np.cos(phi)
                wpmtx[i_g, i_gp] = omega2[i_g, i_gp]
                omega2[i_g, i_gp] *= 1 - 1j * np.tan(phi)

        for i_gp, gpvec in enumerate(g):
            for i_g, gvec in enumerate(g):
                if wtilde2[i_g, i_gp] < 0:
                    # Square of plasma freq < 0: unphysical, see paper for why
                    wtilde2[i_g, i_gp] = 1e24

                    omega2[i_g, i_gp] = wtilde2[i_g, i_gp] * (-1) * epsinv_I[i_g, i_gp]

        assert not np.isnan(wtilde2).any()

        return omega2, wtilde2

    # @pw_logger.time("sigma:sigma_sx_gpp")
    def sigma_sx_gpp(self, dE=0, yielding=True):
        """
        (H.L.) Plasmon Pole Screened Exchange
        ======================================

                                            Omega^2(G,G`)
        SX(E) = M(n,G)*conj(M(m,G`)) * ------------------------ * Vcoul(G`)
                                       (E-E_n1(k-q))^2-wtilde^2
        """

        # Setting einstein summation string for M* M epsinv v
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max - band_index_min + 1

        sigma = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
        if yielding:
            einstr = "kcg, kcp, kcgp, p -> kc"
        else:
            einstr = "kvcg, kvcp, kvcgp, p -> kc"
            #         M*    M     ssx    V    Σ_n=n',c
            # denom = 1/((Ev-Ec)^2 - ω~^2)

        # k   : k-point
        # v   : valence    states (bands)
        # c,d : conduction states (bands)
        # g   : G
        # p   : G'

        l_q = deepcopy(self.qpts.cryst)  # list of q-points in crystal coords

        # set q0 = 0
        l_q[self.qpts.index_q0] *= 0

        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]

        def calculate_ssx(E_bra, E_ket, omega2, wtilde2):
            # shapes for reference:
            #   E_bra: (n_kpts, n_bands_bra or 1)
            #   E_ket: (n_kpts, n_bands_ket)
            added_dimension_to_E_bra = True if len(E_bra.shape) < 2 else False
            if added_dimension_to_E_bra:
                E_bra = E_bra[:, np.newaxis]

            # Calculate (E_nk-E)^2
            E = E_ket + dE

            wxt = self.ryd * (E[:, np.newaxis, :] - E_bra[:, :, np.newaxis])
            denominator_E2 = np.square(wxt)

            # Calculate (E_nk-E)^2 - wtilde^2
            denominator = np.subtract.outer(denominator_E2, wtilde2)

            wtilde = np.sqrt(wtilde2)
            wdiff = np.subtract.outer(wxt, wtilde)
            cden = wdiff
            rden = np.square(cden)
            rden[np.where(rden == 0)] = np.finfo(float).eps
            rden = np.reciprocal(rden)
            delw = wtilde * np.conj(cden) * rden
            delwr = delw * np.conj(delw)
            wdiffr = wdiff * np.conj(wdiff)

            condbrancha = np.logical_and(wdiffr > self.limittwo, delwr < self.limitone)
            condbranchb = delwr > TOLERANCE

            wherebranchnotanotb = np.where(
                np.logical_and(np.logical_not(condbrancha), np.logical_not(condbranchb))
            )

            mask_branchbnota = np.logical_and(np.logical_not(condbrancha), condbranchb)

            cden = denominator
            rden = np.square(cden)
            rden[np.where(rden == 0)] = np.finfo(float).eps
            rden = np.reciprocal(rden)

            ssx = np.einsum(
                "gp, kvcgp, kvcgp -> kvcgp", omega2, np.conj(cden), rden, optimize=True
            )
            ssx = np.nan_to_num(ssx)

            cden_branchbnota = 4 * np.einsum(
                "gp, kvcgp -> kvcgp", wtilde2, (delw + 0.5)
            )
            cden = np.where(mask_branchbnota, cden_branchbnota, cden)

            rden_branchbnota = np.einsum("kvcgp, kvcgp -> kvcgp", cden, np.conj(cden))
            rden_branchbnota[np.where(rden_branchbnota == 0)] = np.finfo(float).eps
            rden_branchbnota = np.reciprocal(rden_branchbnota)

            rden = np.where(mask_branchbnota, rden_branchbnota, rden)

            ssx_branchbnota = -1 * np.einsum(
                "gp, kvcgp, kvcgp, kvcgp -> kvcgp",
                omega2,
                np.conj(cden),
                rden,
                delw,
                optimize=True,
            )
            ssx = np.where(
                mask_branchbnota,
                ssx_branchbnota,
                ssx,
            )

            ssx[wherebranchnotanotb] *= 0

            # Screened exchange cutoff
            np.place(
                ssx,
                np.logical_and(
                    abs(ssx) > 4 * np.abs(epsinv_I), (wxt < 0)[:, :, :, None, None]
                ),
                [0],
            )

            if added_dimension_to_E_bra:
                return ssx[:, 0, ...]
            else:
                return ssx

        # --------------------------------------------------

        if not self.in_parallel:
            iterable = trange(len(l_q), desc="Sigma_SX_GPP")
            proc_q_indices = range(len(l_q))
        else:
            proc_rank = self.comm.Get_rank()
            q_indices = np.arange(self.qpts.numq)
            proc_q_indices = np.array_split(q_indices, self.comm_size)[proc_rank]
            iterable = proc_q_indices

        for i_q in iterable:
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

            # Calculate Omega expression
            # See notes #1 and #2
            omega2, wtilde2 = self.sigma_gpp_omegas(qpt, g, vqg, epsinv_I)

            if yielding:
                for M, E_bra, E_ket in self.matrix_elements(
                    i_minusq, ket_all_bands=True, ret_E=True, yielding=yielding
                ):
                    # shapes for reference:
                    #   M: (n_kpts,n_ket,ngq)
                    #   E_bra: (n_kpts)
                    #   E_ket: (n_kpts, n_bands_ket)

                    ssx = calculate_ssx(E_bra, E_ket, omega2, wtilde2)
                    M = M[..., g_to_mg]
                    sigma_q = np.einsum(
                        einstr,
                        np.conj(M),
                        M,
                        ssx,
                        vqg,
                        optimize=True,
                    )
                    assert not np.isnan(sigma_q).any()

                    sigma += sigma_q

            else:
                # Calculate matrix elements; also returns mean field energy eigenvals for ket states
                M, E_bra, E_ket = next(
                    self.matrix_elements(
                        i_minusq, ket_all_bands=True, ret_E=True, yielding=yielding
                    )
                )

                ssx = calculate_ssx(E_bra, E_ket, omega2, wtilde2)
                M = M[..., g_to_mg]
                sigma_q = np.einsum(
                    einstr,
                    np.conj(M),
                    M,
                    ssx,
                    vqg,
                    optimize=True,
                )
                assert not np.isnan(sigma_q).any()

                sigma += sigma_q

        if self.in_parallel:
            sigma = sum(self.comm.allgather(sigma))

        sigma *= -1 * self.sigma_factor

        if Sigma.autosave:
            np.save(
                self.outdir
                + f"sigma_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sigma_sxgpp{'dE' if np.max(np.abs(dE))!=0 else ''}_{'_'.join([str(q_index) for q_index in proc_q_indices])}",
                sigma,
            )

        return sigma

    # @pw_logger.time("sigma:sigma_ch_gpp")
    def sigma_ch_gpp(self, dE=0, yielding=True):
        """
        Plasmon Pole Coulomb Hole (partial sum)
        =======================================

                                            Omega^2(G,G`)
        CH(E) = M(n,G)*conj(M(m,G`)) * ----------------------------- * Vcoul(G`)
                                    2*wtilde*[E-E_n1(k-q)-wtilde]

        """

        # Setting einstein summation string for M* M epsinv v

        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max
        number_bands_outer = band_index_max - band_index_min + 1

        # if diag:
        ch_gpp = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)
        ch_static = np.zeros((self.n_kpts, number_bands_outer), dtype=complex)

        # einstr = "kvcg, kvcp, gp, kvcgp, p -> kc"
        if yielding:
            einstr = "kcg, kcp, kcgp, p -> kc"
        else:
            einstr = "kvcg, kvcp, kvcgp, p -> kc"
            #         M*    M     ssx    V    Σ_n=n',c
            # denom = 1/((Ev-Ec)^2 - ω~^2)

        # k   : k-point
        # v   : valence    states (bands)
        # c,d : conduction states (bands)
        # g   : G
        # p   : G'

        l_q = deepcopy(self.qpts.cryst)
        l_q[
            self.qpts.index_q0
        ] *= 0  # FIXME: This has been ensured in __init__. Now we can remove it.
        g_cryst = [self.l_gq[i_q].g_cryst for i_q in range(self.qpts.numq)]

        if not self.in_parallel:
            iterable = trange(len(l_q), desc="Sigma_CH_GPP")
            proc_q_indices = range(len(l_q))
        else:
            proc_rank = self.comm.Get_rank()
            q_indices = np.arange(self.qpts.numq)
            proc_q_indices = np.array_split(q_indices, self.comm_size)[proc_rank]
            iterable = proc_q_indices

        # @pw_logger.time("sigma:calculate_sch")
        def calculate_sch(E_bra, E_ket, wtilde2):
            # shapes for reference:
            # E_bra: (n_kpts, n_bands_bra or 1)
            # E_ket: (n_kpts, n_bands_ket)
            added_dimension_to_E_bra = True if len(E_bra.shape) < 2 else False
            if added_dimension_to_E_bra:
                E_bra = E_bra[:, np.newaxis]

            # Calculate (E_nk-E)^2
            E = E_ket + dE
            wxt = self.ryd * (E[:, np.newaxis, :] - E_bra[:, :, np.newaxis])
            denominator_E2 = np.square(wxt)

            # Calculate (E_nk-E)^2 - wtilde^2
            denominator = np.subtract.outer(denominator_E2, wtilde2)

            wtilde = np.sqrt(wtilde2)

            wdiff = np.subtract.outer(wxt, wtilde)
            cden = wdiff
            rden = np.square(cden)
            rden[np.where(rden == 0)] = np.finfo(float).eps
            rden = np.reciprocal(rden)
            delw = wtilde * np.conj(cden) * rden
            delwr = delw * np.conj(delw)
            wdiffr = wdiff * np.conj(wdiff)

            condbrancha = np.logical_and(wdiffr > self.limittwo, delwr < self.limitone)
            condbranchb = delwr > TOLERANCE

            wherebranchbnota = np.where(
                np.logical_and(np.logical_not(condbrancha), condbranchb)
            )
            wherebranchnotanotb = np.where(
                np.logical_and(np.logical_not(condbrancha), np.logical_not(condbranchb))
            )

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

            # there was conj(rden) for some reason
            sch = np.einsum("kvcgp, gp -> kvcgp", delw, (-1) * epsinv_I, optimize=True)
            sch = np.nan_to_num(sch)

            # branch B
            # else if ( delwr .gt. TOL_Zero) then
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
            sch[wherebranchnotanotb] *= 0

            if added_dimension_to_E_bra:
                return sch[:, 0, ...]
            else:
                return sch

        for i_q in iterable:
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

            # Calculate Omega expression
            _, wtilde2 = self.sigma_gpp_omegas(qpt, g, vqg, epsinv_I)

            if yielding:
                for M, E_bra, E_ket in self.matrix_elements(
                    i_minusq,
                    ket_all_bands=True,
                    bra_all_bands=True,
                    ret_E=True,
                    yielding=yielding,
                ):
                    sch = calculate_sch(E_bra, E_ket, wtilde2)
                    M_ = M[..., g_to_mg]
                    ch_gpp += np.einsum(
                        einstr, np.conj(M_), M_, sch, vqg, optimize=True
                    )

                    ch_static += np.einsum(
                        "njk,njm,mk,m->nj",
                        np.conj(M_),
                        M_,
                        epsinv_I,
                        vqg,
                        optimize=True,
                    )
            else:
                # Calculate matrix elements; also returns mean field energy eigenvals for ket states
                M, E_bra, E_ket = next(
                    self.matrix_elements(
                        i_minusq, ket_all_bands=True, bra_all_bands=True, ret_E=True
                    )
                )  # , row_all_bands=True

                sch = calculate_sch(E_bra, E_ket, wtilde2)
                M_ = M[..., g_to_mg]
                ch_gpp += np.einsum(einstr, np.conj(M_), M_, sch, vqg, optimize=True)
                ch_static += np.einsum(
                    "nijk,nijm,mk,m->nj", np.conj(M_), M_, epsinv_I, vqg, optimize=True
                )

        if self.in_parallel:
            ch_gpp = sum(self.comm.allgather(ch_gpp))
            ch_static = sum(self.comm.allgather(ch_static))

        ch_gpp *= 0.5 * self.sigma_factor
        ch_static *= 0.5 * self.sigma_factor

        if Sigma.autosave:
            np.save(
                self.outdir
                + f"sigma_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_sigma_chgpp_{'_'.join([str(q_index) for q_index in proc_q_indices])}",
                ch_gpp,
            )

        return ch_gpp, ch_static

    # ==================================================================
    # Methods to run full calculations

    # @pw_logger.time("sigma:calculate_static_cohsex")
    def calculate_static_cohsex(self):
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max

        # Calculate on-shell QP energy = Emf - Vxc + Sig(Eo)
        vxc_data = self.vxc
        vxc = np.array(vxc_data.vxc)[
            self.slice_l_k, band_index_min - 1 : band_index_max
        ]

        # Emf data
        emf_factor = 1 / ELECTRONVOLT_HART
        emf = np.array([self.l_wfn[i_k].evl for i_k in self.l_k_indices])
        emf = np.array(emf[:, band_index_min - 1 : band_index_max]) * emf_factor
        sigma_x_mat = self.sigma_x()
        if self.print_condition:
            print("Sigma X")
            self.pprint_sigma_mat(sigma_x_mat)

        sigma_sx_mat = self.sigma_sx_static()
        if self.print_condition:
            print("Sigma SX STATIC")
            self.pprint_sigma_mat(sigma_sx_mat)

        sigma_ch_mat = self.sigma_ch_static()
        if self.print_condition:
            print("Sigma CH STATIC")
            self.pprint_sigma_mat(sigma_ch_mat)

        sigma_ch_exact_mat = self.sigma_ch_static_exact()
        if self.print_condition:
            print("Sigma CH EXACT STATIC")
            self.pprint_sigma_mat(sigma_ch_exact_mat)

        sigma_mat_exact = sigma_sx_mat + sigma_ch_exact_mat + sigma_x_mat
        if self.print_condition:
            print("Sig (Exact):")
            self.pprint_sigma_mat(sigma_mat_exact)
            print("Eqp0 (Exact):")
            self.pprint_sigma_mat((sigma_mat_exact + emf - vxc))

        Eqp0_exact = sigma_mat_exact + emf - vxc

        sigma_mat = sigma_sx_mat + sigma_ch_mat + sigma_x_mat
        if self.print_condition:
            print("Sig (Partial):")
            self.pprint_sigma_mat(sigma_mat)
            print("Eqp0 (Partial):")
            self.pprint_sigma_mat((sigma_mat + emf - vxc))

        Eqp0 = sigma_mat + emf - vxc

        print_x = np.real(np.around((sigma_x_mat), 6))
        print_sx = np.real(np.around((sigma_sx_mat), 6))
        print_ch = np.real(np.around((sigma_ch_mat), 6))
        print_exact_ch = np.real(np.around((sigma_ch_exact_mat), 6))

        if self.in_parallel:
            self.comm.Barrier()

        if self.print_condition:
            print()
            for k in range(len(self.l_k_indices)):
                print(
                    "   n         Emf          Eo           X        SX-X          CH         Sig         Vxc        Eqp0        Eqp1         CH`        Sig`       Eqp0`       Eqp1`         Znk"
                )
                for n in range(
                    self.sigmainp.band_index_min - 1,
                    self.sigmainp.band_index_max - self.sigmainp.band_index_min + 1,
                ):
                    print(
                        "{:>4}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}".format(
                            n,  # n
                            np.real(emf[k, n]),  # Emf
                            np.real(emf[k, n]),  # Eo
                            print_x[k, n],  # X
                            print_sx[k, n],  # SX-X
                            print_exact_ch[k, n],  # CH
                            np.real(sigma_mat_exact[k, n]),  # Sig
                            np.real(vxc[k, n]),  # Vxc
                            np.real(Eqp0_exact[k, n]),  # Eqp0
                            np.real(Eqp0_exact[k, n]),  # Eqp1
                            print_ch[k, n],  # CH'
                            np.real(sigma_mat[k, n]),  # Sig'
                            np.real(Eqp0[k, n]),  # Eqp0'
                            np.real(Eqp0[k, n]),  # Eqp1'
                            np.real(1),  # Z
                        )
                    )

        ret_dict = {}
        for k in range(len(self.l_k_indices)):
            ret_dict[self.l_k_indices[k]] = {
                "n": list(
                    range(
                        self.sigmainp.band_index_min - 1,
                        self.sigmainp.band_index_max - self.sigmainp.band_index_min + 1,
                    )
                ),
                "Emf": np.real(emf[k]),
                "Eo": np.real(emf[k]),
                "X": print_x[k],
                "SX-X": print_sx[k],
                "CH": print_exact_ch[k],
                "Sig": np.real(sigma_mat_exact[k]),
                "Vxc": np.real(vxc[k]),
                "Eqp0": np.real(Eqp0_exact[k]),
                "Eqp1": np.real(Eqp0_exact[k]),
                "CH`": print_ch[k],
                "Sig`": np.real(sigma_mat[k]),
                "Eqp0`": np.real(Eqp0[k]),
                "Eqp1`": np.real(Eqp0[k]),
                "Znk": np.ones_like(print_x[k]),
            }
        return ret_dict

    # @pw_logger.time("sigma:sigma_gpp")
    def calculate_gpp(self):
        if self.print_condition and self.comm != None:
            print(
                f"Started calculate_gpp Rank: {self.comm.Get_rank()} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        band_index_min = self.sigmainp.band_index_min
        band_index_max = self.sigmainp.band_index_max

        self.print_condition = (not self.in_parallel) or (
            self.in_parallel and COMM_WORLD.Get_rank() == 0
        )

        # Calculate on-shell QP energy = Emf - Vxc + Sig(Eo)
        # Vxc data
        vxc_data = self.vxc
        vxc = np.array(vxc_data.vxc)[self.slice_l_k][
            :, band_index_min - 1 : band_index_max
        ]

        # Emf data
        emf_factor = 1 / ELECTRONVOLT_HART

        emf = np.array([self.l_wfn[i_k].evl for i_k in self.l_k_indices])
        emf = np.array(emf[:, band_index_min - 1 : band_index_max]) * emf_factor

        # before slicing emf, create dE with the correct shape
        dE = np.zeros_like(emf)
        # Calculate Eqp0
        # ==============

        sigma_x_mat = self.sigma_x()
        if self.print_condition:
            print("Sigma X GPP", flush=True)
            self.pprint_sigma_mat(sigma_x_mat)
        if self.in_parallel:
            self.comm.Barrier()

        sigma_ch_static_mat = self.sigma_ch_static()
        if self.print_condition:
            print("Sigma CH STATIC COHSEX", flush=True)
            self.pprint_sigma_mat(sigma_ch_static_mat)
        if self.in_parallel:
            self.comm.Barrier()

        sigma_ch_exact_mat = self.sigma_ch_static_exact()
        if self.print_condition:
            print("Sigma CH STATIC EXACT", flush=True)
            self.pprint_sigma_mat(sigma_ch_exact_mat)
        if self.in_parallel:
            self.comm.Barrier()

        if self.print_condition:
            print(
                f"Started sigma_sx_gpp {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        sigma_sx_gpp_mat = self.sigma_sx_gpp()
        if self.print_condition:
            print("Sigma SX GPP", flush=True)
            self.pprint_sigma_mat(sigma_sx_gpp_mat)
        if self.in_parallel:
            self.comm.Barrier()

        sigma_ch_gpp_mat, _ = self.sigma_ch_gpp()
        if self.print_condition:
            print("Sigma CH GPP", flush=True)
            self.pprint_sigma_mat(sigma_ch_gpp_mat)
        if self.in_parallel:
            self.comm.Barrier()

        # `Sig` with Static Remainder

        static_remainder_correction = 0.5 * (sigma_ch_exact_mat - sigma_ch_static_mat)

        sigma_mat = (
            sigma_x_mat
            + sigma_sx_gpp_mat
            + sigma_ch_gpp_mat
            + static_remainder_correction
        )
        sigma_mat = np.around((sigma_mat), 6)
        if self.print_condition:
            print("Sig GPP:", flush=True)
            self.pprint_sigma_mat(sigma_mat)

        if self.in_parallel:
            self.comm.Barrier()

        # Unprimed values, such as Eqp0, are evaluated WITH the static remainder
        Eqp0_prime = sigma_mat + emf - vxc - static_remainder_correction

        # Primed values, such as Eqp0`, are evaluated WITHOUT the static remainder
        Eqp0 = sigma_mat + emf - vxc
        if self.print_condition:
            print("Eqp0", flush=True)
            self.pprint_sigma_mat(Eqp0)

        # Calculate Eqp1
        # ==============
        # Assuming:
        # - finite_difference_form = 2:Forward difference
        # - finite_difference_spacing = 1.0
        # Note that sigma.inp docs are incorrect;
        # they mention Ecor, whereas, sigma_hp.log shows Eo, which is Emf for us

        dE[:] = 1.0
        dE *= ELECTRONVOLT_RYD
        # print("dE", dE)
        # FIXME: Remove later, while converting the entire code to Hartree. Presently, the methods in Sigma are in ryd units.
        # dE/=2

        sigma_ch_gpp_mat_2, _ = self.sigma_ch_gpp(dE)
        sigma_sx_gpp_mat_2 = self.sigma_sx_gpp(dE)

        if self.print_condition:
            print("Sigma CH GPP dE")
            self.pprint_sigma_mat(sigma_ch_gpp_mat_2)
            print("Sigma SX GPP dE")
            self.pprint_sigma_mat(sigma_ch_gpp_mat_2)

        dSigdE = (
            (sigma_sx_gpp_mat_2 + sigma_ch_gpp_mat_2)
            - (sigma_sx_gpp_mat + sigma_ch_gpp_mat)
        ) / (dE / ELECTRONVOLT_RYD)
        slope = dSigdE / (1 - dSigdE)
        Z = 1 / (1 - dSigdE)

        if self.print_condition:
            print("Z:")
            self.pprint_sigma_mat(Z)

        # Calculate Eqp0 (with Static Remainder correction)
        # ==================================================

        # Sig with Remainder
        sigma_mat_2 = sigma_sx_gpp_mat_2 + sigma_ch_gpp_mat_2 + sigma_x_mat
        sigma_mat_2 = np.around((sigma_mat_2), 6)

        if self.print_condition:
            print("Sig_2:")
            print(sigma_mat_2.T)

        Eqp1 = Eqp0 + slope * (Eqp0 - emf)
        Eqp1_prime = Eqp0_prime + slope * (Eqp0_prime - emf)
        if self.print_condition:
            print("Eqp1")
            self.pprint_sigma_mat(Eqp1)

        print_x = np.real(np.around((sigma_x_mat), 6))
        print_sx = np.real(np.around((sigma_sx_gpp_mat), 6))
        print_ch = np.real(
            np.around((sigma_ch_gpp_mat + static_remainder_correction), 6)
        )
        print_ch_prime = np.real(np.around((sigma_ch_gpp_mat), 6))
        print_sig_prime = np.real(
            np.around(
                (sigma_x_mat + sigma_sx_gpp_mat + sigma_ch_gpp_mat),
                6,
            )
        )
        if self.print_condition:
            for k in range(len(self.l_k_indices)):
                print(
                    "   n         Emf          Eo           X        SX-X          CH         Sig         Vxc        Eqp0        Eqp1         CH`        Sig`       Eqp0`       Eqp1`         Znk"
                )
                for n in range(
                    self.sigmainp.band_index_min - 1,
                    self.sigmainp.band_index_max - self.sigmainp.band_index_min + 1,
                ):
                    print(
                        "{:>4}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}".format(
                            n,
                            np.real(emf[k, n]),
                            np.real(emf[k, n]),
                            print_x[k, n],
                            print_sx[k, n],
                            print_ch[k, n],
                            np.real(sigma_mat[k, n]),
                            np.real(vxc[k, n]),
                            np.real(Eqp0[k, n]),
                            np.real(Eqp1[k, n]),
                            print_ch_prime[k, n],
                            np.real(print_sig_prime[k, n]),
                            np.real(Eqp0_prime[k, n]),
                            np.real(Eqp1_prime[k, n]),
                            np.real(Z[k, n]),
                        )
                    )
                ret_dict = {}
        for k in range(len(self.l_k_indices)):
            ret_dict[self.l_k_indices[k]] = {
                "n": list(
                    range(
                        self.sigmainp.band_index_min - 1,
                        self.sigmainp.band_index_max - self.sigmainp.band_index_min + 1,
                    )
                ),
                "Emf": np.real(emf[k]),
                "Eo": np.real(emf[k]),
                "X": print_x[k],
                "SX-X": print_sx[k],
                "CH": print_ch[k],
                "Sig": np.real(sigma_mat[k]),
                "Vxc": np.real(vxc[k]),
                "Eqp0": np.real(Eqp0[k]),
                "Eqp1": np.real(Eqp1[k]),
                "CH`": print_ch[k],
                "Sig`": np.real(sigma_mat[k]),
                "Eqp0`": np.real(Eqp0[k]),
                "Eqp1`": np.real(Eqp0[k]),
                "Znk": np.ones_like(print_x[k]),
            }

        return ret_dict


if __name__ == "__main__":
    # dirname = "./test/bgw/"

    dirname = "../../../tests/bgw/silicon/cohsex/"
    outdir = f"./test/tempdir_20230805_113833/"

    # dirname = "../../../tests/benchmark_silicon/si_6_nband272_cohsex/si_6_gw/"
    # outdir = "./test/tempdir_20230806_215443/"

    # Load WFN data

    # from qtm.mpi.comm import qtmconfig.mpi4py_installed, COMM_WORLD
    from qtm.gw.io_bgw import inp
    from qtm.gw.io_bgw.epsmat_read_write import read_mats

    if MPI4PY_INSTALLED and COMM_WORLD.Get_size() > 1:
        in_parallel = True
    else:
        in_parallel = False
    print_condition = (not in_parallel) or (in_parallel and COMM_WORLD.Get_rank() == 0)

    if print_condition:
        print(
            "Sigma script started running : ",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            flush=True,
        )
        print(f"dirname: {dirname}")
        print(f"outdir:  {outdir}")
    print("in_parallel", in_parallel, COMM_WORLD.Get_rank(), flush=True)

    # Read data and create Sigma object

    # Sigma.inp data
    if print_condition:
        print("Reading sigma.inp data", flush=True)
    sigmainp = inp.read_sigma_inp(filename=dirname + "sigma.inp")

    # Epsilon.inp data
    if print_condition:
        print("Reading epsilon.inp data", flush=True)

    epsinp = inp.read_epsilon_inp(filename=dirname + "epsilon.inp")

    # wfn2py
    from qtm.gw.io_bgw.wfn2py import wfn2py

    if print_condition:
        print(f"Reading WFN.h5 from directory: {dirname}", flush=True)

    wfndata = wfn2py(dirname + "WFN.h5")

    if print_condition:
        print(f"Reading WFNq.h5 from directory: {dirname}", flush=True)

    wfnqdata = wfn2py(dirname + "WFNq.h5")

    # To read QTMGW's epsmats:
    epsmats_dirname = outdir
    if print_condition:
        print(
            f"Reading qtm's epsilon matrices from directory: {epsmats_dirname}",
            flush=True,
        )

    l_epsmats_actual = []
    l_epsmats_actual += read_mats(outdir + "eps0mat_qtm.h5")
    l_epsmats_actual += read_mats(outdir + "epsmat_qtm.h5")
    print(
        len(l_epsmats_actual),
        COMM_WORLD.Get_rank(),
        l_epsmats_actual[0].shape,
        flush=True,
    )

    l_epsmats = l_epsmats_actual

    epsmats = []

    rho = inp.read_rho(dirname + "RHO")
    vxc = inp.read_vxc(dirname + "vxc.dat")

    COMM_WORLD.Barrier()
    if print_condition:
        print(
            f"Constructing Sigma object",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            flush=True,
        )

    sigma = Sigma.from_data(
        wfndata=wfndata,
        wfnqdata=wfnqdata,
        sigmainp=sigmainp,
        epsinp=epsinp,
        l_epsmats=l_epsmats,
        rho=rho,
        vxc=vxc,
        outdir=outdir,
    )

    if print_condition:
        print(
            f"Constructed Sigma object",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            flush=True,
        )

    # Run

    if int(sys.argv[1]) == 0:  # Sigma SX STATIC COHSEX
        sigma_sx_static_mat = sigma.sigma_sx_static()
        if print_condition:
            print("Sigma SX STATIC COHSEX", end="\n")
            print(sigma_sx_static_mat, end="\n")
            print(np.around((sigma_sx_static_mat) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 1:  # Sigma SX GPP
        sigma_sx_gpp_mat = sigma.sigma_sx_gpp()
        if print_condition:
            print("Sigma SX GPP", end="\n")
            print(sigma_sx_gpp_mat, end="\n")
            print(np.around((sigma_sx_gpp_mat) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 2:  # Sigma CH GPP
        sigma_ch_gpp_mat, sigma_ch_static_mat = sigma.sigma_ch_gpp()
        if print_condition:
            print("Sigma CH GPP", end="\n")
            print(sigma_ch_gpp_mat, end="\n")
            print(np.around((sigma_ch_gpp_mat) * sigma.sigma_factor, 6))
            print("Sigma CH STATIC COHSEX", end="\n")
            print(sigma_ch_static_mat, end="\n")
            print(np.around((sigma_ch_static_mat) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 3:  # Sigma CH STATIC COHSEX
        sigma_ch_static_mat = sigma.sigma_ch_static()
        if print_condition:
            print("Sigma CH STATIC COHSEX", end="\n")
            print(sigma_ch_static_mat, end="\n")
            print(np.around((sigma_ch_static_mat) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 4:  # Sigma CH STATIC EXACT
        sigma_ch_exact_mat = sigma.sigma_ch_static_exact()
        if print_condition:
            print("Sigma CH STATIC EXACT", end="\n")
            print(sigma_ch_exact_mat, end="\n")
            print(np.around((sigma_ch_exact_mat) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 5:  # Sigma X
        sigma_x_mat = sigma.sigma_x()
        if print_condition:
            print("Sigma X GPP", end="\n")
            print(sigma_x_mat, end="\n")
            print(np.around((sigma_x_mat) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 6:  # Sigma CH GPP dE
        dE = 1.0 / Sigma.ryd

        sigma_ch_gpp_mat_2, _ = sigma.sigma_ch_gpp(dE)
        if print_condition:
            print("Sigma CH GPP dE", end="\n")
            print(sigma_ch_gpp_mat_2, end="\n")
            print(np.around((sigma_ch_gpp_mat_2) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 7:  # Sigma SX GPP dE
        dE = 1.0 / Sigma.ryd

        sigma_sx_gpp_mat_2 = sigma.sigma_sx_gpp(dE)
        if print_condition:
            print("Sigma SX GPP dE", end="\n")
            print(sigma_sx_gpp_mat_2, end="\n")
            print(np.around((sigma_sx_gpp_mat_2) * sigma.sigma_factor, 6))

    if int(sys.argv[1]) == 8:  # Sigma GPP
        sigma.calculate_gpp()

    if int(sys.argv[1]) == 9:  # Sigma Static COHSEX
        sigma.calculate_static_cohsex()

    if print_condition:
        with open(
            outdir
            + f"vcoul_sigma_qtm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "w",
        ) as f:
            f.write(sigma.vcoul.write_vcoul())
        print(
            "Sigma script finished running : ",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
