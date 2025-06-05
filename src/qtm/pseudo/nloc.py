from __future__ import annotations

__all__ = ["NonlocGenerator"]

import numpy as np
from scipy.special import sph_harm
from scipy.linalg import block_diag

from qtm.crystal.basis_atoms import BasisAtoms
from qtm.gspace import GSpace, GkSpace
from qtm.containers import get_WavefunG, WavefunGType
from .upf import UPFv2Data

from qtm.config import NDArray, qtmconfig
from qtm.msg_format import type_mismatch_msg
from qtm.logger import qtmlogger

from qtm.constants import FPI, TPIJ


DEL_Q = 1e-2


def small_x_thr_fn(n):
    if n > 2:
        small_x_thr = 0.05 * 2.5 ** (n - 2) + 0.05 * (n - 3)
    else:
        small_x_thr = 0.05
    return small_x_thr


def semifact(n):
    return np.prod(np.arange(n, 0, -2, dtype="i8"))


def spherical_jn(n: int, x_: np.ndarray) -> np.ndarray:
    # assert isinstance(n, int), type(n)
    assert n >= 0
    if n > 6:
        return NotImplemented
    assert x_.ndim == 1
    # x is expected to be in ascending order
    small_x_thr = small_x_thr_fn(n)
    small_x = np.asarray(small_x_thr, like=x_)
    isplit = np.searchsorted(x_, small_x, "right")

    j_ = np.empty_like(x_)
    if isplit != 0:
        x, j = x_[:isplit], j_[:isplit]
        x2 = x**2
        xn = x**n
        j[:] = (
            xn
            / semifact(2 * n + 1)
            * (
                1
                - (
                    x2
                    / 1
                    / 2
                    / (2 * n + 3)
                    * (
                        1
                        - (
                            x2
                            / 2
                            / 2
                            / (2 * n + 5)
                            * (
                                1
                                - (
                                    x2
                                    / 3
                                    / 2
                                    / (2 * n + 7)
                                    * (1 - x2 / 4 / 2 / (2 * n + 9))
                                )
                            )
                        )
                    )
                )
            )
        )
    if isplit == x_.shape[0]:
        return j_

    x, j = x_[isplit:], j_[isplit:]
    sx = np.sin(x)
    if n == 0:
        j[:] = sx / x
        return j_

    cx = np.cos(x)
    x2 = x**2
    if n == 1:
        j[:] = (sx / x2) - (cx / x)
        return j_
    elif n == 2:
        j[:] = ((3 / x - x) * sx - 3 * cx) / x2
        return j_

    x3 = x * x2
    if n == 3:
        j[:] = ((15 / x - 6 * x) * sx + (x2 - 15) * cx) / x3
        return j_

    x4 = x2 * x2
    x5 = x2 * x3
    if n == 4:
        j[:] = ((105 - 45 * x2 + x4) * sx + (10 * x3 - 105 * x) * cx) / x5
        return j_

    if n == 5:
        j[:] = (
            cx * (-1 - 945 / x4 + 105 / x2) + sx * (945 / x5 - 420 / x3 + 15 / x)
        ) / x
        return j_

    x6 = x3 * x3
    if n == 6:
        j[:] = (
            cx * (-10395 / x5 + 1260 / x3 - 21 / x)
            + sx * (-1 + 10395 / x6 - 4725 / x4 + 210 / x2)
        ) / x
        return j_


class NonlocGenerator:
    """Generates nonlocal part of the pseudopotential of input atoms for a
    given ``GkSpace`` instance

    Parameters
    ----------
    sp : BasisAtoms
        Group of atoms of the same type in unit cell. Must contain
        pseudopotential data
    gspc : GSpace
        G-Space representing the smooth grid for wavefunctions
    """

    @qtmlogger.time("nloc:init")
    def __init__(self, sp: BasisAtoms, gwfn: GSpace):
        # Setting Up
        if sp.ppdata is None:
            raise ValueError(
                f"{BasisAtoms} instance 'sp' does not have "
                f"pseudopotential data i.e 'sp.ppdata' is None."
            )
        if not isinstance(sp.ppdata, UPFv2Data):
            raise NotImplementedError("only 'UPFv2Data' supported")
        if not isinstance(gwfn, GSpace):
            raise ValueError(type_mismatch_msg("gwfn", gwfn, GSpace))
        self.species: BasisAtoms = sp
        self.gwfn: GSpace = gwfn
        self.ecut: float = self.gwfn.ecut / 4

        ppdata: UPFv2Data = self.species.ppdata
        # Radial Mesh specified in Pseudopotential Data
        r = np.asarray(ppdata.r, like=self.gwfn.g_cryst)
        r_ab = np.asarray(ppdata.r_ab, like=self.gwfn.g_cryst)

        # Getting the beta projectors
        self.numbeta = ppdata.number_of_proj
        self.beta_l = ppdata.l_kb_l
        l_kb_rbeta = np.asarray(ppdata.l_kb_rbeta, like=self.gwfn.g_cryst)

        # Defining Function for Simpsons' Integration
        def simpson(f_r: np.ndarray):
            r12 = 1 / 3
            f_times_dr = f_r * r_ab
            # NOTE: Number of radial points specified in UPF File is expected to be odd. Will fail otherwise
            return r12 * np.sum(
                f_times_dr[0:-2:2] + 4 * f_times_dr[1:-1:2] + f_times_dr[2::2]
            )

        # Computing beta projectors in reciprocal space across a fine mesh of q-points for interpolation
        self.numq = int(np.ceil(np.sqrt(2 * self.ecut) / DEL_Q + 4))
        self.q = np.arange(self.numq, like=self.gwfn.g_cryst) * DEL_Q
        self.beta_q = np.empty(
            (self.numbeta, self.numq), dtype="f8", like=self.gwfn.g_cryst
        )
        for iq in range(self.numq):
            q = self.q[iq]
            for ibeta in range(self.numbeta):
                l, rbeta = self.beta_l[ibeta], l_kb_rbeta[ibeta]
                sph_jl_qr = spherical_jn(l, q * r)
                # 4 pi / sqrt(cellvol) prefactor will be multipled later
                self.beta_q[ibeta, iq] = simpson(rbeta * r * sph_jl_qr)

        # Generating mappings between KB projectors and quantum numbers
        self.vkb_idxbeta = np.asarray(
            [ibeta for ibeta, l in enumerate(self.beta_l) for _ in range(-l, l + 1)],
            like=self.gwfn.g_cryst,
        )

        self.vkb_l = np.asarray(
            [l for ibeta, l in enumerate(self.beta_l) for _ in range(-l, l + 1)],
            like=self.gwfn.g_cryst,
        )

        self.vkb_m = np.asarray(
            [
                ((i + 1) // 2) * (-1) ** (i % 2)
                for l in self.beta_l
                for i in range(2 * l + 1)
            ],
            like=self.gwfn.g_cryst,
        )
        self.numvkb = len(self.vkb_l)

        self.dij_beta = np.asarray(ppdata.dij, like=self.gwfn.g_cryst)

    @qtmlogger.time("nloc:gen_vkb_dij")
    def gen_vkb_dij(self, gkspc: GkSpace) -> tuple[WavefunGType, NDArray, WavefunGType]:
        r"""Generates the KB projectors and the dij matrix for the given atomic species.

        Parameters
        ----------
        gkspc : GSpace
            Represents the :math:`\mathbf{G} + \mathbf{k}` vectors that form
            the basis of Wavefunctions
        Returns
        -------
        l_vkb_full : WavefunGType
            list of all beta projectors spanning across all atoms of the species
        dij_full : NDArray
            dij matrix. Expanded for all atoms of the species
        vkb_diag : WavefunGType
            list of diagonal elements of the non-local operator
        """
        # Setting Up: Computing spherical coordinates for all :math:`\mathbf{G}+\mathbf{k}`
        WavefunG = get_WavefunG(gkspc, 1)
        numgk = gkspc.size_g

        gk_cryst = gkspc.gk_cryst
        gk_x, gk_y, gk_z = gkspc.gk_cart
        gk_norm = gkspc.gk_norm
        beta_fac = FPI / np.sqrt(gkspc.reallat_cellvol)

        where_gk_norm_nonzero = np.where(gk_norm > 1e-7)
        theta = np.zeros_like(gk_z)
        theta[where_gk_norm_nonzero] = np.arccos(
            gk_z[where_gk_norm_nonzero] / gk_norm[where_gk_norm_nonzero]
        )
        phi = np.arctan2(gk_y, gk_x)

        l_vkb_atom = gkspc.allocate_array((self.numvkb, numgk))
        idx_gk = np.rint(gk_norm / DEL_Q).astype("i8")
        xmin0 = gk_norm / DEL_Q - idx_gk
        xmin1 = xmin0 - 1
        xmin2 = xmin0 - 2
        xmin3 = xmin0 - 3

        idxvkb = 0
        # Constructing KB Projectors for a single atom
        for idxbeta in range(self.numbeta):
            # Lagrange Interpolation for radial part
            beta_gk = beta_fac * (
                self.beta_q[idxbeta][idx_gk + 0]
                * xmin1
                * xmin2
                * xmin3
                / ((0 - 1) * (0 - 2) * (0 - 3))
                + self.beta_q[idxbeta][idx_gk + 1]
                * xmin0
                * xmin2
                * xmin3
                / ((1 - 0) * (1 - 2) * (1 - 3))
                + self.beta_q[idxbeta][idx_gk + 2]
                * xmin0
                * xmin1
                * xmin3
                / ((2 - 0) * (2 - 1) * (2 - 3))
                + self.beta_q[idxbeta][idx_gk + 3]
                * xmin0
                * xmin1
                * xmin2
                / ((3 - 0) * (3 - 1) * (3 - 2))
            )

            # Applying angular part using spherical harmonics
            l = self.beta_l[idxbeta]
            for abs_m in range(l + 1):
                ylm = sph_harm(abs_m, l, phi, theta)
                if abs_m == 0:
                    l_vkb_atom[idxvkb] = ylm * beta_gk
                else:
                    l_vkb_atom[idxvkb] = (
                        -np.sqrt(2) * (-1) ** abs_m * ylm.imag * beta_gk
                    )
                    idxvkb += 1
                    l_vkb_atom[idxvkb] = (
                        -np.sqrt(2) * (-1) ** abs_m * ylm.real * beta_gk
                    )
                idxvkb += 1

        # Constructing `dij` matrix
        dij_atom = gkspc.allocate_array((self.numvkb, self.numvkb))
        dij_atom[:] = 0
        for i1 in range(self.numvkb):
            for i2 in range(self.numvkb):
                if (
                    self.vkb_l[i1] == self.vkb_l[i2]
                    and self.vkb_m[i1] == self.vkb_m[i2]
                ):
                    dij_atom[i1, i2] = self.dij_beta[
                        self.vkb_idxbeta[i1], self.vkb_idxbeta[i2]
                    ]

        numatoms = self.species.numatoms

        # Generating KB Projectors corresponding to all atoms
        l_vkb_full = WavefunG.empty(numatoms * self.numvkb)
        vkb_diag = WavefunG.zeros(None)
        for iat, pos_cryst in enumerate(self.species.r_cryst.T):
            phase = np.exp(-TPIJ * (pos_cryst @ gk_cryst))
            l_vkb_iat = l_vkb_full[iat * self.numvkb : (iat + 1) * self.numvkb]
            l_vkb_iat.data[:] = phase * l_vkb_atom * (-(1j**self.vkb_l)).reshape(-1, 1)
            vkb_diag += np.sum(l_vkb_iat * (dij_atom @ l_vkb_iat.conj()), axis=0)

        if qtmconfig.gpu_enabled:
            dij_full = block_diag(*[dij_atom.get() for _ in range(numatoms)])
            dij_full = np.asarray(dij_full, like=self.gwfn.g_cryst)
        else:
            dij_full = block_diag(*[dij_atom for _ in range(numatoms)])

        # dij_full = gkspc.allocate_array((self.numvkb * numatoms, self.numvkb * numatoms))
        # dij_full*=0
        # for iat in range(numatoms):
        #     sl = (slice(iat * self.numvkb, (iat + 1) * self.numvkb))
        #     dij_full[sl, sl] = dij_atom

        # for vkb in l_vkb_full:
        #     if np.any(np.isnan(vkb)):
        #         print("Naans in vkb")

        return l_vkb_full, dij_full, vkb_diag


    @qtmlogger.time('nloc:gen_vkb_dij_deriv')
    def gen_vkb_dij_deriv(self, gkspc: GkSpace) -> tuple[WavefunGType, NDArray, WavefunGType]:
        r"""Computes the Nonlocal Operator as a set of Beta Projectors and a
        transformation matrix

        Parameters
        ----------
        gkspc : GSpace
            Represents the :math:`\mathbf{G} + \mathbf{k}` vectors that form
            the basis of Wavefunctions
        Returns
        -------
        l_vkb_full : np.ndarray
            list of all beta projectors spanning across all atoms of the species
        dij_full : np.ndarray
            dij matrix. Expanded for all atoms of the species
        vkb_diag : np.ndarray
            list of diagonal elements of the non-local operator
        """
        # Setting Up: Computing spherical coordinates for all :math:`\mathbf{G}+\mathbf{k}`
        WavefunG = get_WavefunG(gkspc, 1)
        numgk = gkspc.size_g

        gk_cryst = gkspc.gk_cryst
        gk_x, gk_y, gk_z = gkspc.gk_cart
        gk_norm = gkspc.gk_norm
        beta_fac = FPI / np.sqrt(gkspc.reallat_cellvol)

        # theta = np.arccos(
        #     np.divide(gk_z, gk_norm, out=np.zeros_like(gk_z), where=gk_norm > 1e-7)
        # )
        # print('np.where(gk_norm <= 1e-5) :',np.where(gk_norm <= 1e-5)) #debug statement
        where_gk_norm_nonzero = np.where(gk_norm > 1e-7)
        theta = np.zeros_like(gk_z)
        theta[where_gk_norm_nonzero] = np.arccos(gk_z[where_gk_norm_nonzero] / gk_norm[where_gk_norm_nonzero])
        phi = np.arctan2(gk_y, gk_x)


        #the spherical harmonic derivatives are only implemented  along the 
        #projection axis (1,0,0), (0,1,0), (0,0,1). i.e. the normal three dim. coordinate system.
        def dylm(abs_m, l, ipol):
            DIFF=1e-8
            gkcart=gkspc.gk_cart
            dgk_norm=DIFF*gkspc.gk_norm
            dgk_norm_nonzero = np.where(dgk_norm>DIFF**2)
            dgk_norm_inv=np.zeros_like(dgk_norm)
            dgk_norm_inv[dgk_norm_nonzero]=1/dgk_norm[dgk_norm_nonzero]

            gkcart_low=gkcart.copy()
            gkcart_low[ipol]-=dgk_norm
            gkx_low, gky_low, gkz_low = gkcart_low

            gkcart_high=gkcart.copy()
            gkcart_high[ipol]+=dgk_norm
            gkx_high, gky_high, gkz_high = gkcart_high
            
            gk_norm_low=np.sqrt(np.sum(gkcart_low**2,axis=0))
            gk_norm_high=np.sqrt(np.sum(gkcart_high**2,axis=0))

            where_gk_norm_nonzero = np.where(gk_norm_low > 1e-7)
            theta_low = np.zeros_like(gkz_low)
            theta_low[where_gk_norm_nonzero] = np.arccos(gkz_low[where_gk_norm_nonzero] / gk_norm_low[where_gk_norm_nonzero])
            phi_low = np.arctan2(gky_low, gkx_low)

            where_gk_norm_nonzero = np.where(gk_norm_high > 1e-7)
            theta_high = np.zeros_like(gkz_high)
            theta_high[where_gk_norm_nonzero] = np.arccos(gkz_high[where_gk_norm_nonzero] / gk_norm_high[where_gk_norm_nonzero])
            phi_high = np.arctan2(gky_high, gkx_high)

            Ylm_high=sph_harm(abs_m, l, phi_high, theta_high)
            Ylm_low=sph_harm(abs_m, l, phi_low, theta_low)

            diff=(Ylm_high - Ylm_low)*dgk_norm_inv*0.5

            return diff
        

        l_djvkb_atom = gkspc.allocate_array((self.numvkb, numgk))

        l_dyvkb_atom= gkspc.allocate_array((3,self.numvkb, numgk))


        idx_gk = np.rint(gk_norm / DEL_Q).astype("i8")
        xmin0 = gk_norm / DEL_Q - idx_gk
        xmin1 = xmin0 - 1
        xmin2 = xmin0 - 2
        xmin3 = xmin0 - 3

        idxvkb = 0
        # Constructing KB Projectors for a single atom
        for idxbeta in range(self.numbeta):
            # Lagrange Interpolation for radial part
            beta_gk = beta_fac * (
                self.beta_q[idxbeta][idx_gk + 0]
                * xmin1
                * xmin2
                * xmin3
                / ((0 - 1) * (0 - 2) * (0 - 3))
                + self.beta_q[idxbeta][idx_gk + 1]
                * xmin0
                * xmin2
                * xmin3
                / ((1 - 0) * (1 - 2) * (1 - 3))
                + self.beta_q[idxbeta][idx_gk + 2]
                * xmin0
                * xmin1
                * xmin3
                / ((2 - 0) * (2 - 1) * (2 - 3))
                + self.beta_q[idxbeta][idx_gk + 3]
                * xmin0
                * xmin1
                * xmin2
                / ((3 - 0) * (3 - 1) * (3 - 2))
            )
            djbeta_gk = beta_fac * (
                self.beta_q[idxbeta][idx_gk + 0]
                * (xmin1*xmin2+xmin2*xmin3+xmin3*xmin1)
                / ((0 - 1) * (0 - 2) * (0 - 3))
                + self.beta_q[idxbeta][idx_gk + 1]
                * (xmin3*xmin2 +xmin0*xmin2 + xmin0*xmin3)
                / ((1 - 0) * (1 - 2) * (1 - 3))
                + self.beta_q[idxbeta][idx_gk + 2]
                * (xmin1*xmin3 +xmin0*xmin1 + xmin0*xmin3)
                / ((2 - 0) * (2 - 1) * (2 - 3))
                + self.beta_q[idxbeta][idx_gk + 3]
                * (xmin2*xmin1 +xmin0*xmin1 + xmin0*xmin2)
                / ((3 - 0) * (3 - 1) * (3 - 2))
            )/DEL_Q
            # Applying angular part using spherical harmonics
            l = self.beta_l[idxbeta]
            for abs_m in range(l + 1):
                ylm = sph_harm(abs_m, l, phi, theta)
                dy_lm=np.array([dylm(abs_m, l, ipol) for ipol in [0,1,2]])
                if abs_m == 0:
                     l_djvkb_atom[idxvkb] = ylm * djbeta_gk
                     l_dyvkb_atom[:,idxvkb,:]=dy_lm*beta_gk
                else:
                     l_djvkb_atom[idxvkb] =  -np.sqrt(2) * (-1) ** abs_m * ylm.imag * djbeta_gk
                     l_dyvkb_atom[:,idxvkb,:]=-np.sqrt(2) * (-1) ** abs_m * dy_lm.imag * beta_gk
                     idxvkb += 1
                     l_djvkb_atom[idxvkb] = -np.sqrt(2) * (-1) ** abs_m * ylm.real * djbeta_gk
                     l_dyvkb_atom[:,idxvkb,:]=-np.sqrt(2) * (-1) ** abs_m * dy_lm.real * beta_gk
                idxvkb += 1

        numatoms = self.species.numatoms

        # Generating KB Projectors corresponding to all atoms
        l_djvkb_full = WavefunG.empty(numatoms * self.numvkb)
        l_dyvkbx_full= WavefunG.empty(numatoms * self.numvkb)
        l_dyvkby_full= WavefunG.empty(numatoms * self.numvkb)
        l_dyvkbz_full= WavefunG.empty(numatoms * self.numvkb)
        for iat, pos_cryst in enumerate(self.species.r_cryst.T):
            phase = np.exp(-TPIJ * (pos_cryst @ gk_cryst))
            l_djvkb_iat = l_djvkb_full[iat*self.numvkb: (iat+1)*self.numvkb]
            l_dyvkbx_iat= l_dyvkbx_full[iat*self.numvkb: (iat+1)*self.numvkb]
            l_dyvkby_iat= l_dyvkby_full[iat*self.numvkb: (iat+1)*self.numvkb]
            l_dyvkbz_iat= l_dyvkbz_full[iat*self.numvkb: (iat+1)*self.numvkb]
            l_djvkb_iat.data[:] = phase * l_djvkb_atom * (-(1j**self.vkb_l)).reshape(-1, 1)
            l_dyvkbx_iat.data[:]=phase * l_dyvkb_atom[0] * (-(1j**self.vkb_l)).reshape(-1, 1)
            l_dyvkby_iat.data[:]=phase * l_dyvkb_atom[1] * (-(1j**self.vkb_l)).reshape(-1, 1)
            l_dyvkbz_iat.data[:]=phase * l_dyvkb_atom[2] * (-(1j**self.vkb_l)).reshape(-1, 1)

        l_dyvkb_full=(l_dyvkbx_full, l_dyvkby_full, l_dyvkbz_full)

        # dij_full = gkspc.allocate_array((self.numvkb * numatoms, self.numvkb * numatoms))
        # dij_full*=0
        # for iat in range(numatoms):
        #     sl = (slice(iat * self.numvkb, (iat + 1) * self.numvkb))
        #     dij_full[sl, sl] = dij_atom

        # for vkb in l_vkb_full:
        #     if np.any(np.isnan(vkb)):
        #         print("Naans in vkb")

        return l_djvkb_full, l_dyvkb_full
