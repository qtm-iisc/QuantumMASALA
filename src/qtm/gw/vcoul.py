from copy import deepcopy
from functools import lru_cache
from itertools import chain

import numpy as np
from numpy.random import MT19937, RandomState
from scipy import spatial
from tqdm import trange

from qtm.constants import RYDBERG_HART
from qtm.gspace import GkSpace, GSpace
from qtm.config import MPI4PY_INSTALLED
from qtm.gw.core import QPoints, sort_cryst_like_BGW

if MPI4PY_INSTALLED:
    from mpi4py import MPI




class Vcoul:
    r"""Vcoul Generator Class

    Provides cell-averaging technique whereby the value of the interaction
    for the given q-point (for a q → 0 in particular) can be replaced by the average
    of v(q+G) in the volume the q-point represents.

    This average can be made to include the q-dependence of the inverse dielectric function
    if W is the final quantity of relevance for the application such as
    the evaluation of W for the self-energy.

    TODO: Recheck documentation when completed.

    Attributes
    ----------
    gspace: GSpace
        The largest GSpace object, contains the entire GSpace grid
        Will be used to extract smaller spheres for each qpt
    qpts: QPoints
        qpoints provided in inp files
    gspace_q: List[GSpaceQpt]
        List of GSpaceQpt objects, one for each qpt in qpts.
    vcoul: List[float]
        Stores vcoul values: [index_qpt, index_gspace_pt]
    avgcut: float
        If \|q+G\|^2 < avgcut, calculate <1/(q+G)^2> (i.e. average). Otherwise, calculate 1/(q+G)^2
    bare_coulomb_cutoff: float
        Cutoff provided in inp files, for gspace of qpts: \|q+G\|^2 < bare_coulomb_cutoff

    Methods
    -------
    write_vcoul()
        Write vcoul data to file. Conforms to ``vcoul`` format of BGW.
    v_bare(i_q)
        Calculate vcoul value for qpt with index ``i_q``.
    v_minibz_sphere()
        Calculate vcoul averaged over minibz ball for qpt = (0,0,0).
    v_minibz_sphere_shifted(centre_cryst)
        Calculate vcoul averaged over minibz ball for \|qpt\| > R_minibz_sphere
    v_minibz_montecarlo()
        Calculate vcoul averaged using montecarlo over minibz for qpt = (0,0,0)
    v_minibz_montecarlo_hybrid()
        Calculate analytically in the sphere inscribed in minibz,
        but add the contribution from the left-over region using montecarlo.
    """

    N_SAMPLES = 2.5e6
    N_SAMPLES_COARSE = 2.5e5
    SEED = 5000
    TOL_SMALL = 1e-5

    def __init__(
        self,
        gspace: GSpace,
        qpts: QPoints,
        bare_coulomb_cutoff: float,
        avgcut: float = TOL_SMALL,
        bare_init=True,  # To save time, by default
        parallel=True,
    ) -> None:
        r"""Init Vcoul object

        Parameters
        ----------
        gspace: GSpace
            The largest GSpace object, contains the entire GSpace grid
            Will be used to extract smaller spheres for each qpt
        qpts: QPoints
            Constructed from qpoints provided in inp files
        bare_coulomb_cutoff: float
            Cutoff specified in inp files
        avgcut: float
            If \|q+G\|^2 < avgcut, calculate <1/(q+G)^2>. Otherwise, calculate 1/(q+G)^2
        """
        self.gspace = gspace
        self.qpts = qpts
        self.avgcut = avgcut
        self.vcoul = []

        self.bare_coulomb_cutoff = bare_coulomb_cutoff
        self.l_gspace_q = [
            GkSpace(
                gwfn=gspace,
                k_cryst=self.qpts.cryst[i_q],
                ecutwfn=bare_coulomb_cutoff * RYDBERG_HART,
            )
            for i_q in range(self.qpts.numq)
        ]

        self.in_parallel = False
        self.comm = None
        self.comm_size = None
        if parallel:
            if MPI4PY_INSTALLED:
                self.comm = MPI.COMM_WORLD
                self.comm_size = self.comm.Get_size()
                if self.comm_size > 1:
                    self.in_parallel = True

        self.calculate_vcoul(bare=bare_init)  # averaging_func=None)

        return

    def __repr__(self):
        string = f"""Vcoul:
        * gspace = {self.gspace}
        * qpts = {self.qpts}
        * bare_coulomb_cutoff = {self.bare_coulomb_cutoff}
        * avgcut = {self.avgcut}
        * l_gspace_q = {type(self.l_gspace_q)} of length {len(self.l_gspace_q)}
        * vcoul = {type(self.vcoul)} of length {len(self.vcoul)}
        * N_SAMPLES = {Vcoul.N_SAMPLES}
        * N_SAMPLES_COARSE = {Vcoul.N_SAMPLES_COARSE}
        * SEED = {Vcoul.SEED}
        """
        if Vcoul.SEED != 5000:
            string += (
                f"\nWARNING: Vcoul.SEED={self.SEED} is not the same as BGW's default."
            )
        if Vcoul.N_SAMPLES != 2.5e6:
            string += f"\nWARNING: Vcoul.N_SAMPLES={self.N_SAMPLES} is not the same as BGW's default."
        if Vcoul.N_SAMPLES_COARSE != 2.5e5:
            string += f"\nWARNING: Vcoul.N_SAMPLES_COARSE={self.N_SAMPLES_COARSE} is not the same as BGW's default."
        return string

    # HELPER FUNCTIONS :
    def fixwings(
        self,
        epsinv,
        i_q,
        q0,
        q0flag: bool,
        wcoul0: float,
        vcoul: float,
        random_sample: bool,
    ):
        r"""Modify epsmat to make it compatible with W averaging

        Parameters

        epsinvmat : np.ndarray
            Generated by epsilon.py/f90

        Returns

        fixed_epsinv : np.array
            With head and wing elements fixed.


        Notes from BGW, for reference

            ! MODULE: fixwings_m
            !
            !> Rescale epsmat to make it compatible with W averaging
            !
            ! DESCRIPTION:
            !> This module contains routines to rescale the wings and \b head of the
            !! epsilon matrix so that later we can compute <W> = <epsinv(q) * v(q)>.
            !! Note that <W> should take into consideration the analytical form of epsmat(q)
            !! and v(q) for small q, for each type of truncation/screening.
            !! Despite its name, this routine fixes both the wings and \b head of the
            !! GPP epsmat. The goal is to rescale epsmat so that we get the appropriate
            !! W averaging, i.e., \f$ W_0 = \varepsilon^{-1}(q) v(q) \f$.

            From sigma_main.f90:
            ! Fix wing divergence for semiconductors and graphene
            ! This should really be done for all "\|q+G\| < avgcut" - but for now,
            ! it is done if "\|q\| < avgcut and G=0"
        """

        fixed_epsinv = deepcopy(epsinv)
        # NOTE: Double check epsmat is to be used as-is or transpose, because of Fortran vs C index ordering

        # Older code
        def fixwing(arr, i_q, q0len, wingtype: str, oneoverq: float, vcoul: float):
            """Given a sub-matrix from epsinv and the wingtype, fix it.

            This nested function is exactly like the fixwings of BGW."""

            if wingtype == "wing":  # G!=0 and G'=0
                if q0flag:
                    return arr * 0
                else:
                    return arr * oneoverq / (vcoul * q0len)
            elif wingtype == "wingprime":  # G=0 and G'!=0
                if q0flag:
                    return arr * 0
                else:
                    return arr * oneoverq * q0len / (8 * np.pi)
            elif wingtype == "head":  # G=0 and G'=0
                if q0flag:
                    return np.ones_like(arr) * wcoul0 / vcoul

            return arr

        # Find G=0 for the given qpt's gvec list,
        # in order to calcualte oneoverq for that G=0
        index_G0 = np.argmin(
            np.linalg.norm(
                self.l_gspace_q[i_q].g_cryst.T[
                    sort_cryst_like_BGW(self.l_gspace_q[i_q].gk_cryst, self.l_gspace_q[i_q].gk_norm2)
                    # self.l_gspace_q[i_q].gk_indices_tosorted
                ],
                axis=1,
            )
        )
        # print(f"i_q: {i_q}, index_G0: {index_G0}")
        # Slice excluding index_G0: Not Implemented. Implement later

        oneoverq = self.oneoverq[i_q]

        # Head
        fixed_epsinv[index_G0, index_G0] = fixwing(
            fixed_epsinv[index_G0, index_G0],
            i_q,
            q0,
            wingtype="head",
            oneoverq=oneoverq,
            vcoul=vcoul,
        )
        # save head, because it will be temporarily modified in absence of good slice
        head = np.copy(fixed_epsinv[index_G0, index_G0])

        # Wing
        fixed_epsinv[:, index_G0] = fixwing(
            fixed_epsinv[:, index_G0],
            i_q,
            q0,
            wingtype="wingprime",
            oneoverq=oneoverq,
            vcoul=vcoul,
        )

        # Wing'
        fixed_epsinv[index_G0, :] = fixwing(
            fixed_epsinv[index_G0, :],
            i_q,
            q0,
            wingtype="wing",
            oneoverq=oneoverq,
            vcoul=vcoul,
        )

        # Restore Head
        fixed_epsinv[index_G0, index_G0] = head

        return fixed_epsinv

    @lru_cache()
    def mt19937_samples(self, nsamples=N_SAMPLES):
        rs = RandomState(self.SEED)
        mt19937 = MT19937()
        mt19937.state = rs.get_state()

        # Discard first 2500000*3 random numbers. Don't delete the next line.
        dump = mt19937.random_raw(int(self.N_SAMPLES) * 3)
        return mt19937.random_raw(3 * int(nsamples))

    # CORE FUNCTIONS :

    def v_bare(self, i_q, averaging_func=None):
        r"""Calculate Coulomb potential (Reciprocal space), given index of q-point, i.e. 8*pi/\|q+G\|^2"""

        # Get sorted \|q+G\|^2
        gqq = self.l_gspace_q[i_q].gk_norm2

        # Calculate bare Coulomb interaction
        vqg = np.ones_like(gqq) * np.nan
        vqg[np.nonzero(gqq)] = 8 * np.pi * np.reciprocal(gqq[np.nonzero(gqq)])

        return vqg

    # MINIBZ AVERAGING :

    def v_minibz_sphere(self):
        """Analytical integral of bare Vcoul within sphere of same volume as mini-bz

        V0_sphere = 3 * 8 * pi / q_cutoff^2
        """
        recvol = self.gspace.recilat.cellvol
        q_cutoff = (recvol / len(self.qpts.cryst) * 3 / (4 * np.pi)) ** (1 / 3)

        return 3 * 8 * np.pi / q_cutoff**2

    def v_minibz_sphere_shifted(self, centre_cryst, q_cutoff=None):
        r"""Averaged coulomb potential within a shifted sphere, with same volume as mini-bz

        Constraint: \|centre_cryst\| > R_minibz_sphere
        Can be used to calculate vcoul averaged over minibz ball for qpt != (0,0,0)
        Reason d'etre: Vcoul(q)=#/q^2 varies significantly even for qpt close to  0.

        Parameters
        ----------
        centre_cryst : np.ndarray
            The centre of sphere in crystal coordinates. shape is (3)
        """

        recvol = self.gspace.recilat.cellvol
        if q_cutoff is None:
            q_cutoff = (recvol / len(self.qpts.cryst) * 3 / (4 * np.pi)) ** (1 / 3)

        R = q_cutoff
        d = np.sqrt(self.gspace.recilat.norm2(centre_cryst))

        if d == 0:
            return np.Inf

        numerator = (2 * np.pi) * ((R**2 / d - d) * np.arctanh(R / d) + R)
        denominator = 4 * np.pi * R**3 / 3

        return 8 * np.pi * numerator / denominator

    def v_minibz_montecarlo(self, nsamples=N_SAMPLES, shift_vec_cryst=None, seed=None):
        """Naive montecarlo, returns answer in Rydberg a.u."""
        if seed != None:
            np.random.seed(seed)

        nsamples = int(nsamples)

        # Generate nsamples number of points within nearest neighbourhood
        grid_a = len(self.qpts.cryst) ** (-1 / 3)
        samples_crys = np.random.uniform(
            low=-grid_a, high=grid_a, size=nsamples * 3
        ).reshape((nsamples, 3))

        # Crystal coords to Cartesian coords (a.u.)
        samples_cart = self.gspace.recilat.cryst2cart(samples_crys.T).T

        # Calculate k_cutoff (a.u.)
        recvol = self.gspace.recilat.cellvol
        k_cutoff = (recvol / len(self.qpts.cryst) * 3 / (4 * np.pi)) ** (1 / 3)

        # Since qpts are all positive, we copy the qpts 8 times to create (2*n_q-1)^3 sized grid around q=0
        full_qpts = self.get_gamma_nbhd_qpts()

        # Crystal to Cart:
        full_kpts = self.gspace.recilat.cryst2cart(full_kpts.T).T
        if shift_vec_cryst is None:
            shift_vec_cart = self.gspace.recilat.cryst2cart(np.zeros((1, 3), float).T).T
        else:
            shift_vec_cart = self.gspace.recilat.cryst2cart(
                shift_vec_cryst.reshape(-1, 3).T
            ).T

        # Filter points in grid that are within 3*k_cutoff, as a rough approximation,
        # not necessarily true, but for now, while testing
        # and hence their wigner-seitz plane can cut the sphere with radius k_cutoff
        # nbhd converted to CARTESIAN coords (in a.u.), because we need norm.

        ZERO_TOL = 1e-5

        norm_arr = np.linalg.norm(full_kpts, axis=1)
        nbhd = full_kpts[
            np.where(np.all([norm_arr > ZERO_TOL, norm_arr <= (3 * k_cutoff)], axis=0))
        ]

        v_sum = 0
        oneoverq_sum = 0
        n_cnt = 0

        for pt in samples_cart:
            d = np.linalg.norm(pt)
            if not np.any(
                np.linalg.norm(nbhd - pt, axis=1) < d
            ):  # self.closer_than_d(pt, nbhd, d):
                r = np.linalg.norm(pt + shift_vec_cart)
                v_sum += r ** (-2)
                oneoverq_sum += 1 / r
                n_cnt += 1

        v_avg = (8 * np.pi) * v_sum / n_cnt
        oneoverq_avg = (8 * np.pi) * oneoverq_sum / n_cnt

        if not shift_vec_cryst is None:
            return v_avg, oneoverq_avg
        return v_avg

    def get_gamma_nbhd_qpts(self):
        """Generate a neighbourhood of q=0, that will be used to determine minibz, while sampling points later."""
        # Since qpts are all positive, we copy the qpts 8 times to create (2*n_q-1)^3 sized grid around q=0.
        full_qpts = []
        l_signs = np.array([(1-2*i,1-2*j,1-2*k) for i in (0,1) for j in (0,1) for k in (0,1)])
        
        # Equivalent, older version
        # l_signs = np.array([ [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])

        # list of q-points
        l_k = deepcopy(self.qpts.cryst)
        l_k[self.qpts.index_q0] = [0, 0, 0]
        for s in l_signs:
            full_qpts += list(np.multiply(s, l_k))
        full_qpts = np.unique(full_qpts, axis=0)

        # Filter points in grid that are within 2*k_cutoff,
        # and hence their wigner-seitz plane can cut the sphere with radius k_cutoff
        # nbhd converted to CARTESIAN coords (a.u.), because we need norm.
        normsq_arr = self.gspace.recilat.norm2(full_qpts.T)
        q_cutoff = np.sqrt(np.min(normsq_arr[np.where(normsq_arr > self.TOL_SMALL)])) / 2

        # 32.0D0 * PI_D**2 * SQRT(q0sph2) / ( 8.0D0 * PI_D**3 / (celvol * dble(nfk)) )
        nbhd_crys = full_qpts[np.where(normsq_arr <= (8 * q_cutoff) ** 2)]
        nbhd_cart = self.gspace.recilat.cryst2cart(nbhd_crys.T).T

        return q_cutoff, nbhd_cart

    def v_minibz_montecarlo_hybrid(self, shift_vec_cryst=None):
        """Return average Coulomb in miniBZ. (Rydberg a.u.)

        Notes
        -----
        - For non-q0 vectors, BGW ooptimizes calculation by keeping the number of points sampled propto 1/e_kin:
            ! FHJ: for spherical integration regions, one can make the error per MC
            ! integration ~const. by choosing the number of points such that N ~ 1/ekinx.
            ! This is because, in 3D, error = sigma/N^{3/2}, and sigma ~ 1/ekinx^{3/2}
            ! If we fix the number of points such that N(ekinx=4*q0sph2) = nmc_coarse,
        """

        # Add G-space shift, useful for non-q0 points.
        if shift_vec_cryst is None:
            shift_vec_cart = self.gspace.recilat.cryst2cart(np.zeros((1, 3), float).T).T
        else:
            shift_vec_cart = self.gspace.recilat.cryst2cart(
                shift_vec_cryst.reshape(-1, 3).T
            ).T

        q_cutoff, nbhd_cart = self.get_gamma_nbhd_qpts()

        shift_length = np.linalg.norm(shift_vec_cart)
        if shift_length < self.TOL_SMALL:
            v_sph = (
                32
                * np.pi**2
                * q_cutoff
                / (8 * np.pi**3 / (self.gspace.reallat_cellvol * self.qpts.numq))
            )
            nsamples = self.N_SAMPLES
        else:
            v_sph = self.v_minibz_sphere_shifted(shift_vec_cryst, q_cutoff=q_cutoff)

            # From BGW minibzaverage_3d:
            # nn2 = idnint(nmc_coarse * 4d0 * q0sph2 / length_qk)
            # nn2 = max(1, min(nn2, nn))
            # Apparently idnint is equivalent to np.round
            nsamples = np.round(
                self.N_SAMPLES_COARSE * 4.0 * q_cutoff**2 / shift_length**2
            )
            nsamples = max(1, min(self.N_SAMPLES, nsamples))
            q_cutoff = 0

        nsamples = int(nsamples)

        sample_factor = 1 / (2**32 * self.qpts.numq ** (1 / 3))
        arr_pt_crys = (
            self.mt19937_samples()[: 3 * nsamples].reshape(-1, 3) * sample_factor
        )
        arr_pt_cart = self.gspace.recilat.cryst2cart(arr_pt_crys.T).T

        tree = spatial.KDTree(nbhd_cart)
        arr_index_closest_nbhd_cart = tree.query(arr_pt_cart)[1]
        arr_pt_cart -= np.take(nbhd_cart, arr_index_closest_nbhd_cart, axis=0)
        arr_length = np.linalg.norm(arr_pt_cart, axis=1)
        arr_length_shifted = np.linalg.norm(arr_pt_cart + shift_vec_cart, axis=1)

        where_length_gt_qcutoff = np.where(arr_length > q_cutoff)
        arr_oneoverlength = np.reciprocal(arr_length_shifted)
        n_corr = len(where_length_gt_qcutoff[0])
        v_corr = np.sum(np.square(arr_oneoverlength[where_length_gt_qcutoff]))
        oneoverq_corr = np.sum(arr_oneoverlength)

        # Had to have the following conditional, because the formula used for
        # v_q0 averaged in BGW takes the smaller volume of inscribed sphere into account
        # whereas, the formula used by us does not.
        if shift_length < self.TOL_SMALL:
            v_sph_factor = 1
        else:
            v_sph_factor = 1 - n_corr / nsamples

        return (
            v_sph * v_sph_factor + (8 * np.pi) * v_corr / nsamples,
            (8 * np.pi) * oneoverq_corr / nsamples,
        )

    def oneoverq_minibz_montecarlo(self, shift_vec_cryst=None):

        # Add G-space shift, useful for non-q0 points.
        if shift_vec_cryst is None:
            shift_vec_cart = self.gspace.recilat.cryst2cart(np.zeros((1, 3), float).T).T
        else:
            shift_vec_cart = self.gspace.recilat.cryst2cart(
                shift_vec_cryst.reshape(-1, 3).T
            ).T

        # Since qpts are all positive, we copy the qpts 8 times to create (2*n_q-1)^3 sized grid around q=0
        q_cutoff, nbhd_cart= self.get_gamma_nbhd_qpts()

        oneoverq_corr = 0

        shift_length = np.linalg.norm(shift_vec_cart)

        # From BGW minibzaverage_3d:
        # nn2 = idnint(nmc_coarse * 4d0 * q0sph2 / length_qk)
        # nn2 = max(1, min(nn2, nn))
        # Apparently idnint is equivalent to np.round
        nsamples = (
            np.round(self.N_SAMPLES_COARSE * 4.0 * q_cutoff**2 / shift_length**2)
            if shift_length > 0
            else np.nan
        )
        nsamples = int(max(1, min(self.N_SAMPLES, nsamples)))

        sample_factor = 1 / (2**32 * self.qpts.numq ** (1 / 3))
        arr_pt_crys = (
            self.mt19937_samples()[: 3 * nsamples].reshape(-1, 3) * sample_factor
        )
        arr_pt_cart = self.gspace.recilat.cryst2cart(arr_pt_crys.T).T

        tree = spatial.KDTree(nbhd_cart)
        arr_index_closest_nbhd_cart = tree.query(arr_pt_cart)[1]
        arr_pt_cart -= np.take(nbhd_cart, arr_index_closest_nbhd_cart, axis=0)
        arr_length_shifted = np.linalg.norm(arr_pt_cart + shift_vec_cart, axis=1)

        oneoverq_corr = np.sum(np.reciprocal(arr_length_shifted))

        return (8 * np.pi) * oneoverq_corr / nsamples

    def oneoverq_minibz_sphere(self, qlen=None):
        """Analytical integral of bare Vcoul within sphere of same volume as mini-bz

        V0_sphere = 3 * 8 * pi / q_cutoff^2
        Units: Rydberg a.u.
        """
        if qlen is None:
            recvol = self.gspace.recilat.cellvol
            q_cutoff = (recvol / len(self.qpts.cryst) * 3 / (4 * np.pi)) ** (1 / 3)
        else:
            q_cutoff = qlen

        return 12 * np.pi / q_cutoff

    def load_vcoul(self, filename):
        """Read `vcoul` file. 
        
        Data format is `q1_cryst  q2_cryst  q3_cryst  G_1  G_2  G_3  vcoul`."""
        
        vcoul_data = np.genfromtxt(filename)

        return (
            np.array(vcoul_data[:, :3]),
            np.array(vcoul_data[:, 3:6]),
            np.array(vcoul_data[:, 6]),
        )

    def write_vcoul(self, i_q_list=None):
        """Write Vcoul in BGW format

        Format:
        q1_cryst    q2_cryst    q3_cryst        G1      G2    G3       Vcoul
        0.00000000  0.00000000  0.00100000       0      0      0       0.22077974E+08
        0.00000000  0.00000000  0.00100000      -1     -1     -1       0.22092680E+02

        Reference snippet from BGW ``vcoul_generator.f90``:
            write(19,'(3f12.8,1x,3i7,1x,e20.8)') &
                qvec_mod(:),gvec%components(:,isrtq(ig)),vcoul(ig)
        ...
        """
        # FIXME: Needs Fortran-like 0.xxxx kind of format for vcoul column.
        res = ""
        if i_q_list is None:
            i_q_list = range(self.qpts.numq)
        for i_q in i_q_list:
            for i_g, gvec in enumerate(self.l_gspace_q[i_q].g_cryst.T):
                if i_q == self.qpts.index_q0:
                    qvec = self.qpts.q0vec
                else:
                    qvec = self.qpts.cryst[i_q]

                res += f"{qvec[0]:>11.8f} {qvec[1]:>11.8f} {qvec[2]:>11.8f}  {gvec[0]:>6} {gvec[1]:>6} {gvec[2]:>6}       {self.vcoul[i_q][i_g]:<.7E}\n"
        return res

    def set_vcoul_and_oneoverq(self, vcoul, oneoverq):
        self.vcoul = vcoul
        self.oneoverq = oneoverq

    def get_vcoul_and_oneoverq(self):
        return self.vcoul, self.oneoverq

    def bcast_vcoul_and_oneoverq(self, comm=None):
        if comm is not None:
            comm.Barrier()
            self.vcoul = comm.bcast(self.vcoul, root=0)
            self.oneoverq = comm.bcast(self.oneoverq, root=0)
        return

    # METHODS :
    #@pw_logger.time('Vcoul:calculate_vcoul_single_qpt')
    def calculate_vcoul_single_qpt(
        self, i_q, averaging_func=None, bare=False, random_avg=True
    ):
        if self.qpts.index_q0 == i_q:
            qvec = np.zeros_like(self.qpts.cryst[i_q])
        else:
            qvec = self.qpts.cryst[i_q]
        qvec_cryst = qvec
        qvec_cart = self.gspace.recilat.cryst2cart(qvec_cryst.reshape(-1, 3).T).T
        qlength = np.linalg.norm(qvec_cart)

        if bare:
            vqg = self.v_bare(i_q, averaging_func)

            # oneoverq calculation
            if self.qpts.index_q0 == i_q:
                qvec = np.zeros_like(self.qpts.cryst[i_q])
            else:
                qvec = self.qpts.cryst[i_q]

            oneoverq = 8 * np.pi / qlength if qlength > 0 else np.nan
            # self.oneoverq.append(oneoverq)

        else:
            if self.qpts.index_q0 == i_q:
                qvec = np.zeros_like(self.qpts.cryst[i_q])
            else:
                qvec = self.qpts.cryst[i_q]
            vqg = np.zeros(self.l_gspace_q[i_q].g_cryst.shape[1])

            for i_g, gvec in enumerate(self.l_gspace_q[i_q].g_cryst.T):
                shift_vec_cryst = qvec + self.l_gspace_q[i_q].g_cryst.T[i_g]

                if random_avg:
                    res = self.v_minibz_montecarlo_hybrid(
                        shift_vec_cryst=shift_vec_cryst
                    )
                    if res[0].ndim == 1:
                        vqg[i_g] = res[0][0]
                    else:
                        vqg[i_g] = res[0]
                    
                    
                else:
                    vqg[i_g] = self.v_minibz_sphere_shifted(qvec + gvec)

            oneoverq = self.oneoverq_minibz_montecarlo(shift_vec_cryst=qvec_cryst)

        return vqg, oneoverq

    # @pw_logger.time("Vcoul:calculate_vcoul")
    def calculate_vcoul(
        self, averaging_func=None, bare=False, random_avg=True, parallel=True
    ):
        """Populate the vcoul list vcoul[i_q][i_g].

        The onus of using appropriate vcoul averaging function
        to modify vcoul entries is on the user.

        TODO: Implement bare_cutoff and averaged_cutoff.
        """

        self.vcoul = []
        self.oneoverq = []

        if not (self.in_parallel and parallel):
            for i_q in trange(self.qpts.numq, desc="Vcoul calculation for qpts"):
                vqg, oneoverq = self.calculate_vcoul_single_qpt(
                    i_q, averaging_func, bare, random_avg
                )
                self.vcoul.append(vqg)
                self.oneoverq.append(oneoverq)
        else:
            proc_vcoul = []
            proc_oneoverq = []
            proc_rank = self.comm.Get_rank()

            q_indices = np.arange(self.qpts.numq)
            proc_q_indices = np.array_split(q_indices, self.comm_size)[proc_rank]

            for i_q in proc_q_indices:
                print(i_q, end=" ", flush=True)
                vqg, oneoverq = self.calculate_vcoul_single_qpt(
                    i_q, averaging_func, bare, random_avg
                )
                proc_vcoul.append(vqg)
                proc_oneoverq.append(oneoverq)

            self.comm.Barrier()
            gathered_oneoverq = self.comm.allgather(proc_oneoverq)
            gathered_vcoul = self.comm.allgather(proc_vcoul)
            self.vcoul = list(chain.from_iterable(gathered_vcoul))
            self.oneoverq = list(chain.from_iterable(gathered_oneoverq))
            self.comm.Barrier()
        return

    def calculate_fixedeps(
        self, epsinv, i_q, wcoul0=None, random_sample=False, fix_nonq0=True
    ):
        """Populate the vcoul list wcoul[i_q][i_g].

        Use fixwings for q or q'==0
        Sample wcoul as <eps_head(q) * vcoul(q)> over minibz.
        """
        sort_order = sort_cryst_like_BGW(self.l_gspace_q[i_q].gk_cryst, self.l_gspace_q[i_q].gk_norm2)
        vcoul = self.vcoul[i_q][sort_order][0]
        if wcoul0 is None:
            wcoul0 = epsinv[0, 0] * vcoul

        if i_q == self.qpts.index_q0:
            q0len = np.sqrt(
                self.l_gspace_q[self.qpts.index_q0].cryst_to_norm2(self.qpts.q0vec)
            )
            fixed_epsinv = self.fixwings(
                epsinv, i_q, q0len, True, wcoul0, deepcopy(vcoul), random_sample=True
            )
        elif fix_nonq0:
            qlen = np.sqrt(self.l_gspace_q[i_q].cryst_to_norm2(self.qpts.cryst[i_q]))
            fixed_epsinv = self.fixwings(
                epsinv, i_q, qlen, False, wcoul0, deepcopy(vcoul), random_sample
            )
        else:
            fixed_epsinv = epsinv.copy()

        return fixed_epsinv
