from copy import deepcopy

from typing import Dict, List
import numpy as np
from tqdm import trange

# from quantum_masala.gw.core import GSpaceQpt
from quantum_masala.core import GSpace, GkSpace

from quantum_masala.constants import RYDBERG_HART

from quantum_masala.gw.core import QPoints

from numpy.random import MT19937
from numpy.random import RandomState


TOL_SMALL = 1e-5


class Vcoul:
    """Vcoul Generator Class

    Equivalent to ``Common/vcoul_generator`` in BGW.

    Provides cell-averaging technique whereby the value of the interaction
    for the given q-point (for a q → 0 in particular) can be replaced by the average
    of v(q+G) in the volume the q-point represents.

    This average can be made to include the q-dependence of the inverse dielectric function
    if W is the final quantity of relevance for the application such as
    the evaluation of W for the self-energy.

    FIXME: Recheck documentation when completed.

    Attributes
    ----------
    gspaces: GSpace
        The largest GSpace object, contains the entire GSpace grid
        Will be used to extract smaller spheres for each qpt
    qpts: QPoints
        qpoints provided in inp files
    gspace_q: List[GSpaceQpt]
        List of GSpaceQpt objects, one for each qpt in qpts.
    vcoul: List
        Stores vcoul values: [index_qpt, index_gspace_pt]
    wcoul: List
        Stores wcoul (screened coulomb) values: [index_qpt, index_gspace_pt]
    averaging_mode_name: Dict
    averaging_mode: int
    w_ready: bool
        Whether wcoul has been populated
    avgcut: float
        If |q+G|^2 < avgcut, calculate <1/(q+G)^2> (i.e. average). Otherwise, calculate 1/(q+G)^2
    bare_coulomb_cutoff: float
        Cutoff provided in inp files, for gspace of qpts: |q+G|^2 < bare_coulomb_cutoff

    Methods
    -------
    write_vcoul()
        Write vcoul data to file. Conforms to ``vcoul`` format of BGW.
    v_bare(i_q)
        Calculate vcoul value for qpt with index ``i_q``.
    v_minibz_sphere()
        Calculate vcoul averaged over minibz ball for qpt = (0,0,0).
    v_minibz_sphere_shifted(centre_cryst)
        Calculate vcoul averaged over minibz ball for |qpt| > R_minibz_sphere
    v_minibz_montecarlo()
        Calculate vcoul averaged using montecarlo over minibz for qpt = (0,0,0)
    v_minibz_montecarlo_hybrid()
        Calculate vcoul averaged using montecarlo over minibz for qpt = (0,0,0)

    Notes
    -----
    - Truncation has not been implemented yet.
    - Performance related note: We assume that this class is being used to
        find vcoul for all qpts. Assumption used to deciding whether or not to
        construct GSpaceQpt.
    - No smart "hack" has been provided. Ref to BGW vcoul_generator:
        !! we "hack" v to make it consistent with W, otherwise the partition (SX-X) + X
        !! would not be correct for metals. We never perform these "hacks" on v(q)
        !! if peinf%jobtypeeval=0.:
    """

    # Consider using enum for truncflag and avgflag
    N_SAMPLES = 2.5e4
    N_SAMPLES_COARSE = 2.5e4
    SEED = 5000
    print(f"Warning! Sigma.SEED = {SEED}")

    def __init__(
        self,
        gspace: GSpace,
        qpts: QPoints,
        bare_coulomb_cutoff: float,
        avgcut: float = TOL_SMALL,
        bare_init = True, # To save time by default
    ) -> None:
        """Init Vcoul object

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
            If |q+G|^2 < avgcut, calculate <1/(q+G)^2>. Otherwise, calculate 1/(q+G)^2
        """
        self.gspace = gspace
        self.qpts = qpts
        self.avgcut = avgcut
        self.vcoul = []
        
        self.bare_coulomb_cutoff = bare_coulomb_cutoff
        self.l_gspace_q = [
            GkSpace(
                gspc=gspace,
                k_cryst=self.qpts.cryst[i_q],
                ecutwfc=bare_coulomb_cutoff * RYDBERG_HART,
            )
            for i_q in range(self.qpts.numq)
        ]

        self.calculate_vcoul(bare=bare_init)#averaging_func=None)

        return

    def __repr__(self):
        string = f"""Vcoul:
            * gspace = {self.gspace}
            * qpts = {self.qpts}
            * bare_coulomb_cutoff = {self.bare_coulomb_cutoff}
            * avgcut = {self.avgcut}
            * l_gspace_q = {type(self.l_gspace_q)} of length {len(self.l_gspace_q)}
            * vcoul = {type(self.vcoul)} of length {len(self.vcoul)}
        """
        return string

    # HELPER FUNCTIONS :

    def closer_than_d(self, pt, l_pts, d):
        """Check if distance of point `pt` to some point in `l_pts` is less than `d`."""
        for r in l_pts:
            if np.linalg.norm(r - pt) <= d:
                return True
        return False

    def ws_sphere_inout(self, pt, l_pts, centre, radius):
        """Correction for spherical approximation for montecarlo in wigner-seitz. [All Cartesian coords]
            `ws` means "Wigner Seitz"

        Returns
        -------
        * 0 if in both or not in either
        * 1 if in ws but not in sphere
        * -1 if in sphere but not in ws
        """

        d_centre = np.linalg.norm(pt - centre)
        d_nbhd, closest_pt, closest_i = self.closest_pt_dst(pt, l_pts)
        # d_nbhd = np.min(np.linalg.norm(pt[None,:]-l_pts), axis=1)

        in_sph = d_centre <= radius
        in_ws = d_centre <= d_nbhd  # AS: Why not divide by 2?

        if (in_sph and in_ws) or not (in_sph or in_ws):
            return 0
        if not in_sph and in_ws:
            return 1
        if in_sph and not in_ws:
            return -1
        return None

    def closest_pt_dst(self, pt, l_pts):
        """Find distance to the closest point from the neighbourhood and the distance from it. [All Cartesian coords]

        Parameters
        ----------
        pt: List[float]
        l_pts: List[List[float]]

        Returns
        -------
        min_dist: float
            The minimum distance
        min_pt: List[float]
            The corresponding nearest point
        min_i: int
            Index of nearest point in the given list.

        Notes
        -----
        Consider using `scipy.spatial.kdtree` instead of this function and `closer_than_d`.
        """
        if len(l_pts) < 1:
            return None, None, None

        min_dist = np.linalg.norm(pt - l_pts[0])
        min_pt = l_pts[0]
        min_i = 0

        for i, r in enumerate(l_pts):
            dist = np.linalg.norm(r - pt)
            if min_dist >= dist:
                min_dist, min_pt, min_i = dist, r, i

        return min_dist, min_pt, min_i

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
        """Rescale epsmat to make it compatible with W averaging

        Parameters
        ----------
        epsinvmat : np.ndarray
            Generated by epsilon.py/f90

        Returns
        -------
        fixed_epsinv : np.array
            With head and wing elements fixed.


        Notes from BGW
        --------------
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
            ! This should really be done for all "|q+G| < avgcut" - but for now,
            ! it is done if "|q| < avgcut and G=0"
        """

        fixed_epsinv = deepcopy(epsinv)
        # NOTE: Double check epsmat is to be used as-is or transpose, because of Fortran vs C index ordering

        # Older code
        def fixwing(arr, i_q, q0len, wingtype: str, oneoverq: float, vcoul: float):
            """Given a sub-matrix from epsinv and the wingtype, fix it.
            This nested function is exactly like th efixwings of BGW."""

            print(f"wcoul0: {wcoul0}, vcoul: {vcoul}, q0len: {q0len}, oneoverq: {oneoverq}")

            # if wingtype=="head":
            #     print("oneoverq  ", oneoverq)
            #     print("q0len     ", q0len)
            #     print("8 PI/q0len", 8*np.pi/q0len)
            #     print(self.qpts.cryst[i_q])

            # match wingtype:
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
        index_G0 = np.argmin(np.linalg.norm(self.l_gspace_q[i_q].g_cryst.T[self.l_gspace_q[i_q].gk_indices_tosorted], axis=1))
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

    # CORE FUNCTIONS :

    def v_bare(self, i_q, averaging_func=None):
        """Calculate Coulomb potential (Reciprocal space), given index of q-point
        i.e. 8*pi/|q+G|^2
        """

        # Get sorted |q+G|^2
        gqq = self.l_gspace_q[i_q].norm2
        # gqq = self.l_gspace_q[i_q].shifted_norm2

        # Calculate bare Coulomb interaction
        vqg = 8 * np.pi * np.reciprocal(gqq)
        # vqg[np.isnan(vqg)] = self.v0_sph()

        # if averaging_func != None and i_q==self.qpts.index_q0:
        #     zero_index = np.argmin(gqq)
        #     vqg[zero_index] = averaging_func()
        #     # vqg[np.isnan(vqg)] = averaging_func()

        return vqg

    # MINIBZ AVERAGING :

    def v_minibz_sphere(self):
        """Analytical integral of bare Vcoul within sphere of same volume as mini-bz
        V0_sphere = 3 * 8 * pi / q_cutoff^2
        Units: Rydberg a.u.
        """
        recvol = self.gspace.recilat.cellvol
        q_cutoff = (recvol / len(self.qpts.cryst) * 3 / (4 * np.pi)) ** (1 / 3)

        return 3 * 8 * np.pi / q_cutoff**2

    def v_minibz_sphere_shifted(self, centre_cryst, q_cutoff=None):
        """Averaged coulomb potential within a shifted sphere, with same volume as mini-bz
        Constraint: |centre_cryst| > R_minibz_sphere
        Can be used to calculate vcoul averaged over minibz ball for qpt != (0,0,0)
        Reason d'etre: Vcoul(q)=#/q^2 is significantly varying for qpt ~ 0.

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
        # d = np.linalg.norm(centre_cryst @ self.gspace.recilat.recvec * self.gspace.recilat.tpiba)

        if d == 0:
            return np.Inf

        numerator = (2 * np.pi) * ((R**2 / d - d) * np.arctanh(R / d) + R)
        denominator = 4 * np.pi * R**3 / 3

        return 8 * np.pi * numerator / denominator

    def v_minibz_montecarlo(self, nsamples=N_SAMPLES, shift_vec_cryst=None, seed=None):
        """Naive montecarlo, returns answer in a.u."""
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
        full_kpts = []
        l_signs = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ]
        )
        l_k = deepcopy(self.qpts.cryst)
        l_k[self.qpts.index_q0] = [0, 0, 0]

        for s in l_signs:
            full_kpts += list(np.multiply(s, l_k))
        full_kpts = np.unique(full_kpts, axis=0)
        # full_kpts_shifted = full_kpts + shift_vec_cryst[None,:]

        # Crystal to Cart:
        # full_kpts = self.gspace.recilat.primvec @ full_kpts * crystal.blat
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
            if not self.closer_than_d(pt, nbhd, d):
                r = np.linalg.norm(pt + shift_vec_cart)
                v_sum += r ** (-2)
                oneoverq_sum += 1 / r
                n_cnt += 1

        v_avg = (8 * np.pi) * v_sum / n_cnt
        oneoverq_avg = (8 * np.pi) * oneoverq_sum / n_cnt

        if not shift_vec_cryst is None:
            return v_avg, oneoverq_avg
        return v_avg

    def v_minibz_montecarlo_hybrid(self, shift_vec_cryst=None):
        """Return average Coulomb in miniBZ. (a.u.)

        Notes
        -----
        - For non-q0 vectors, BGW ooptimizes calculation by keeping the number of points sampled propto 1/e_kin:
            ! FHJ: for spherical integration regions, one can make the error per MC
            ! integration ~const. by choosing the number of points such that N ~ 1/ekinx.
            ! This is because, in 3D, error = sigma/N^{3/2}, and sigma ~ 1/ekinx^{3/2}
            ! If we fix the number of points such that N(ekinx=4*q0sph2) = nmc_coarse,
        """

        # print(shift_vec_cryst, end="\t")

        # Relative error calculation: 1/sqrt(nsamples)
        # nsamples = int(nsamples)

        # Init random number generator
        # seed=5000 is in accordance with BGW
        # print(f"Warning: seed is {self.SEED}")
        rs = RandomState(self.SEED)
        mt19937 = MT19937()
        mt19937.state = rs.get_state()

        # Discard first 2500000*3 random numbers.
        dump = mt19937.random_raw(int(self.N_SAMPLES) * 3) / 2**32
        # dump = mt19937.random_raw(2500000*3)/2**32
        # print("discarding initial random variables:", mt19937.random_raw(2500000*3)/2**32)

        # NOTE: This was older version
        #       The newer version, takes the in-sphere of the miniBZ.
        #
        # calculate k_cutoff, which is radius of sphere with same volume as miniBZ at q=0
        #   math: 4pi/3 k_c^3 = recvol/nqpts

        # Add G-space shift, useful for non-q0 points.
        if shift_vec_cryst is None:
            shift_vec_cart = self.gspace.recilat.cryst2cart(np.zeros((1, 3), float).T).T
        else:
            shift_vec_cart = self.gspace.recilat.cryst2cart(
                shift_vec_cryst.reshape(-1, 3).T
            ).T

        # Since qpts are all positive, we copy the qpts 8 times to create (2*n_q-1)^3 sized grid around q=0
        full_qpts = []
        l_signs = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ]
        )

        # list of q-points
        l_k = deepcopy(self.qpts.cryst)
        l_k[self.qpts.index_q0] = [0, 0, 0]
        for s in l_signs:
            full_qpts += list(np.multiply(s, l_k))
        full_qpts = np.unique(full_qpts, axis=0)

        # print(f"full_qpts {len(full_qpts)}")
        # print(full_qpts)

        # Filter points in grid that are within 2*k_cutoff,
        # and hence their wigner-seitz plane can cut the sphere with radius k_cutoff
        # nbhd converted to CARTESIAN coords (a.u.), because we need norm.
        # ZERO_TOL = 1e-5

        normsq_arr = self.gspace.recilat.norm2(full_qpts.T)

        q_cutoff = np.sqrt(np.min(normsq_arr[np.where(normsq_arr > TOL_SMALL)])) / 2
        # print("q0sph2:", q_cutoff**2)

        # 32.0D0 * PI_D**2 * SQRT(q0sph2) / ( 8.0D0 * PI_D**3 / (celvol * dble(nfk)) )

        nbhd_crys = full_qpts[np.where(normsq_arr <= (8 * q_cutoff) ** 2)]

        # print(nbhd_crys)
        nbhd_cart = self.gspace.recilat.cryst2cart(nbhd_crys.T).T

        v_corr = 0
        oneoverq_corr = 0
        n_corr = 0

        shift_length = np.linalg.norm(shift_vec_cart)
        if shift_length < TOL_SMALL:
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
            # print("nn2 = ", nsamples)
            q_cutoff = 0

        nsamples = int(nsamples)

        for _ in range(nsamples):  # , desc=str(shift_vec_cryst)):
            pt_crys = mt19937.random_raw(3) / 2**32
            pt_crys /= self.qpts.numq ** (1 / 3)

            pt_cart = self.gspace.recilat.cryst2cart(pt_crys.T).T

            # Shift pt from q-grid zone to WS cell
            index_closest_nbhd_cart = np.argmin(
                np.linalg.norm(nbhd_cart - pt_cart, axis=1)
            )
            pt_cart -= nbhd_cart[index_closest_nbhd_cart]
            pt_crys -= nbhd_crys[index_closest_nbhd_cart]

            length = np.linalg.norm(pt_cart)
            length_shifted = np.linalg.norm(pt_cart + shift_vec_cart)
            oneoverlength = 1 / length_shifted
            oneoverq_corr += oneoverlength
            if length > q_cutoff:
                # pt is in ws but not in sph, needs to be counted
                n_corr += 1
                v_corr += oneoverlength**2

        # Had to have the following conditional, because the formula used for
        # v_q0 averaged in BGW takes the smaller volume of inscribed sphere into account
        # whereas, the formula used by us does not.
        if shift_length < TOL_SMALL:
            v_sph_factor = 1
        else:
            v_sph_factor = 1 - n_corr / nsamples

        return (
            v_sph * v_sph_factor + (8 * np.pi) * v_corr / nsamples,
            (8 * np.pi) * oneoverq_corr / nsamples,
        )  # + oneoverqsph * (1-n_corr/nsamples)

    def oneoverq_minibz_montecarlo(
        self, shift_vec_cryst=None
    ):  # , hybrid_for_non0 = False):
        # print(shift_vec_cryst, end="\t")

        # Relative error calculation: 1/sqrt(nsamples)
        # nsamples = int(nsamples)

        # Init random number generator
        # seed=5000 is in accordance with BGW
        # print(f"Warning: seed is {self.SEED}")
        rs = RandomState(self.SEED)
        mt19937 = MT19937()
        mt19937.state = rs.get_state()

        # Discard first 2500000*3 random numbers.
        # dump = mt19937.random_raw(2500000*3)/2**32
        dump = mt19937.random_raw(int(self.N_SAMPLES) * 3) / 2**32
        dump=dump # to avoid commenting out the dumping line, because dump is not used
        # print("discarding initial random variables:", mt19937.random_raw(2500000*3)/2**32)

        # NOTE: This was older version
        #       The newer version, takes the in-sphere of the miniBZ.
        #
        # calculate k_cutoff, which is radius of sphere with same volume as miniBZ at q=0
        #   math: 4pi/3 k_c^3 = recvol/nqpts

        # Add G-space shift, useful for non-q0 points.
        if shift_vec_cryst is None:
            shift_vec_cart = self.gspace.recilat.cryst2cart(np.zeros((1, 3), float).T).T
        else:
            shift_vec_cart = self.gspace.recilat.cryst2cart(
                shift_vec_cryst.reshape(-1, 3).T
            ).T

        # Since qpts are all positive, we copy the qpts 8 times to create (2*n_q-1)^3 sized grid around q=0
        full_qpts = []
        l_signs = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [1, -1, -1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, -1, -1],
            ]
        )

        # list of q-points
        l_k = deepcopy(self.qpts.cryst)
        l_k[self.qpts.index_q0] = [0, 0, 0]
        for s in l_signs:
            full_qpts += list(np.multiply(s, l_k))
        full_qpts = np.unique(full_qpts, axis=0)

        # print(f"full_qpts {len(full_qpts)}")
        # print(full_qpts)

        # Filter points in grid that are within 2*k_cutoff,
        # and hence their wigner-seitz plane can cut the sphere with radius k_cutoff
        # nbhd converted to CARTESIAN coords (a.u.), because we need norm.
        # ZERO_TOL = 1e-5

        normsq_arr = self.gspace.recilat.norm2(full_qpts.T)

        q_cutoff = np.sqrt(np.min(normsq_arr[np.where(normsq_arr > TOL_SMALL)])) / 2
        # print("q0sph2:", q_cutoff**2)

        # 32.0D0 * PI_D**2 * SQRT(q0sph2) / ( 8.0D0 * PI_D**3 / (celvol * dble(nfk)) )

        nbhd_crys = full_qpts[np.where(normsq_arr <= (8 * q_cutoff) ** 2)]

        # print(nbhd_crys)
        nbhd_cart = self.gspace.recilat.cryst2cart(nbhd_crys.T).T

        oneoverq_corr = 0

        shift_length = np.linalg.norm(shift_vec_cart)
        # print("shift_length", shift_length)
        # if True:
        oneoverqsph = 8 * np.pi / shift_length
        # print("oneoverq_minibz_sphere", oneoverqsph)
        # v_sph = 32 * np.pi**2 * q_cutoff / (8*np.pi**3/(self.gspace.reallat_cellvol * self.qpts.numq))
        # From BGW minibzaverage_3d:
        # nn2 = idnint(nmc_coarse * 4d0 * q0sph2 / length_qk)
        # nn2 = max(1, min(nn2, nn))
        
        # Apparently idnint is equivalent to np.round
        nsamples = np.round(self.N_SAMPLES_COARSE * 4.0 * q_cutoff**2 / shift_length**2)
        nsamples = max(1, min(self.N_SAMPLES, nsamples))
        # Older line:
        # nsamples = self.N_SAMPLES

        twopowminus32_times_numq_cuberoot = 2**-32 * self.qpts.numq ** (-1 / 3)

        nsamples = int(nsamples)
        for _ in range(nsamples):
            pt_crys = mt19937.random_raw(3) * twopowminus32_times_numq_cuberoot

            pt_cart = self.gspace.recilat.cryst2cart(pt_crys.T).T

            # Shift pt from q-grid zone to WS cell
            index_closest_nbhd_cart = np.argmin(
                np.linalg.norm(nbhd_cart - pt_cart, axis=1)
            )
            pt_cart -= nbhd_cart[index_closest_nbhd_cart]

            length_shifted = np.linalg.norm(pt_cart + shift_vec_cart)
            oneoverlength = 1 / length_shifted
            oneoverq_corr += oneoverlength

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

    # READ AND LOAD `vcoul` file :

    def load_vcoul(self, filename):
        """Data format is q_x q_  q_z  G_x  G_y  G_z  vcoul"""
        # vcoul = np.array(read_txt("./QE_data/control_scripts/vcoul"))
        vcoul_data = np.genfromtxt(filename)

        return (
            np.array(vcoul_data[:, :3]),
            np.array(vcoul_data[:, 3:6]),
            np.array(vcoul_data[:, 6]),
        )

    def write_vcoul(self, i_q_list=None):
        """Write Vcoul
        Format:
        q1_cryst    q2_cryst    q3_cryst        G1      G2    G3       Vcoul
        0.00000000  0.00000000  0.00100000       0      0      0       0.22077974E+08
        0.00000000  0.00000000  0.00100000      -1     -1     -1       0.22092680E+02

        From BGW ``vcoul_generator.f90``:
            write(19,'(3f12.8,1x,3i7,1x,e20.8)') &
                qvec_mod(:),gvec%components(:,isrtq(ig)),vcoul(ig)
        ...
        """
        # FIXME: Needs Fortran-like 0.xxxx kind of format for vcoul column.
        if i_q_list is None:
            i_q_list = range(self.qpts.numq)
        for i_q in i_q_list:
            for i_g, gvec in enumerate(self.l_gspace_q[i_q].g_cryst.T):
                if i_q == self.qpts.index_q0:
                    qvec = self.qpts.q0vec
                else:
                    qvec = self.qpts.cryst[i_q]
                # gvec = self
                print(
                    f"{qvec[0]:>11.8f} {qvec[1]:>11.8f} {qvec[2]:>11.8f}  {gvec[0]:>6} {gvec[1]:>6} {gvec[2]:>6}       {self.vcoul[i_q][i_g]:<.8E}"
                )

    # METHODS :

    def calculate_vcoul(self, averaging_func=None, bare=False, random_avg=True):
        """Populate the vcoul list vcoul[i_q][i_g].
        The onus of using appropriate vcoul averaging function
        to modify vcoul entries is on the user.

        TODO: Implement bare_cutoff and averaged_cutoff.
        """
        self.vcoul = []
        self.oneoverq = []
        for i_q in trange(self.qpts.numq, desc="Vcoul calculation for qpts"):
            # print(f"Calculate vcoul: i_q\t{i_q}")

            if self.qpts.index_q0 == i_q:
                qvec = np.zeros_like(self.qpts.cryst[i_q])
            else:
                qvec = self.qpts.cryst[i_q]
            qvec_cryst = qvec
            qvec_cart = self.gspace.recilat.cryst2cart(qvec_cryst.reshape(-1, 3).T).T
            qlength = np.linalg.norm(qvec_cart)

            if bare: #This was in the fixwings version in 4x4x4-results branch of newpygw
            # if bare or qlength**2 >= self.avgcut:
                # print("bare vcoul")
                self.vcoul.append(self.v_bare(i_q, averaging_func))

                # oneoverq calculation
                if self.qpts.index_q0 == i_q:
                    qvec = np.zeros_like(self.qpts.cryst[i_q])
                else:
                    qvec = self.qpts.cryst[i_q]

                # Deprecated: Oneoverq is calcualted only for G=0
                # oneoverq_qg = np.zeros(self.l_gspace_q[i_q].cryst.shape[1])
                # for i_g, gvec in enumerate(self.l_gspace_q[i_q].cryst.T):
                #     shift_vec_cryst=qvec+self.l_gspace_q[i_q].cryst.T[i_g]
                #     shift_vec_cart = self.gspace.recilat.cryst2cart(shift_vec_cryst.reshape(-1,3).T).T
                #     shift_length = np.linalg.norm(shift_vec_cart)
                #     oneoverq_qg[i_g] = 8*np.pi/shift_length

                # for i_g, gvec in enumerate(self.l_gspace_q[i_q].cryst.T):
                oneoverq = 8 * np.pi / qlength
                self.oneoverq.append(oneoverq)

            else:
                # print("averaged vcoul")
                if self.qpts.index_q0 == i_q:
                    qvec = np.zeros_like(self.qpts.cryst[i_q])
                else:
                    qvec = self.qpts.cryst[i_q]
                vqg = np.zeros(self.l_gspace_q[i_q].g_cryst.shape[1])

                for i_g, gvec in enumerate(self.l_gspace_q[i_q].g_cryst.T):
                    shift_vec_cryst = qvec + self.l_gspace_q[i_q].g_cryst.T[i_g]
                    # shift_vec_cart = self.gspace.recilat.cryst2cart(shift_vec_cryst.reshape(-1,3).T).T

                    if random_avg:
                        vqg[i_g], _ = self.v_minibz_montecarlo_hybrid(
                            shift_vec_cryst=shift_vec_cryst
                        )  # , hybrid_for_non0=True)
                    else:
                        vqg[i_g] = self.v_minibz_sphere_shifted(qvec + gvec)
                    # print(f"vqg[{i_g}] = {vqg[i_g]}")

                self.vcoul.append(vqg)

                oneoverq = self.oneoverq_minibz_montecarlo(shift_vec_cryst=qvec_cryst)
                # print(f"oneoverq: {oneoverq}")

                # print(f"qvec: {qvec_cryst}, oneoverq: {oneoverq}")
                self.oneoverq.append(oneoverq)

            # self.write_vcoul([i_q])

        return

    def calculate_fixedeps(
        self, epsinv, i_q, wcoul0=None, random_sample=False, fix_nonq0=True
    ):
        """Populate the vcoul list wcoul[i_q][i_g].
        Use fixwings for q or q'==0
        Sample wcoul as <eps_head(q) * vcoul(q)> over minibz
        """
        # raise NotImplementedError("Work in progress")
        # print(self.vcoul[0][0])

        # if self.qpts.index_q0 == i_q:
        # qvec = np.zeros_like(self.qpts.cryst[i_q])
        # vcoul = self.vcoul[i_q][0]
        # else:
        #     qvec = self.qpts.cryst[i_q]
        #     qvec_cryst=qvec
        #     qvec_cart = self.gspace.recilat.cryst2cart(qvec_cryst.reshape(-1,3).T).T
        #     qlength = np.linalg.norm(qvec_cart)
        #     vcoul = 8*np.pi/qlength**2
        vcoul = self.vcoul[i_q][self.l_gspace_q[i_q].gk_indices_tosorted][0]
        self.write_vcoul([i_q])
        # print(vcoul)
        if wcoul0 is None:
            wcoul0 = epsinv[0,0] * vcoul

        # wcoul0 = epsinv[0,0] * self.vcoul[i_q][0]
        print(wcoul0)

        if i_q == self.qpts.index_q0:
            q0len = np.sqrt(self.l_gspace_q[self.qpts.index_q0].cryst_to_norm2(self.qpts.q0vec))
            fixed_epsinv = self.fixwings(epsinv, i_q, q0len, True, wcoul0, deepcopy(vcoul), random_sample=True)
        elif fix_nonq0:
            qlen = np.sqrt(self.l_gspace_q[i_q].cryst_to_norm2(self.qpts.cryst[i_q]))
            fixed_epsinv = self.fixwings(epsinv, i_q, qlen, False, wcoul0, deepcopy(vcoul), random_sample)
        else:
            fixed_epsinv = epsinv.copy()

        return fixed_epsinv


if __name__ == "__main__":
    from quantum_masala.gw.io_bgw.wfn2py import wfn2py
    from quantum_masala.gw.io_bgw.inp import read_epsilon_inp
    from quantum_masala.gw.core import QPoints

    wfndata = wfn2py("./test/bgw/WFN.h5")
    epsdata = read_epsilon_inp()

    qpts = QPoints.from_cryst(wfndata.kpts.recilat, epsdata.is_q0, *epsdata.qpts)

    vcoul = Vcoul(wfndata.grho, qpts, epsdata.epsilon_cutoff)
