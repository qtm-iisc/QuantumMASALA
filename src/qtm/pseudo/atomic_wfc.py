from __future__ import annotations

__all__ = ["AtwfcGenerator"]

import numpy as np
from scipy.special import sph_harm
from scipy.linalg import block_diag

from qtm.crystal.basis_atoms import BasisAtoms, spdf_to_l, str_to_nl
from qtm.gspace import GSpace, GkSpace
from qtm.containers import get_WavefunG, WavefunGType
from .upf import UPFv2Data
from .nloc import spherical_jn, DEL_Q

from qtm.config import NDArray, qtmconfig
from qtm.msg_format import type_mismatch_msg
from qtm.logger import qtmlogger

from qtm.constants import FPI, TPIJ
from functools import lru_cache


class AtwfcGenerator: 
    @qtmlogger.time("atwfc:init")
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

        # Getting the atomic wavefunctions
        self.numwfc = ppdata.number_of_wfc
        self.nl = ppdata.nl
        self.lchi = np.asarray([spdf_to_l(nlstr[-1]) for nlstr in self.nl], like=self.gwfn.g_cryst)
        rchi_ = np.asarray([ppdata.atwfc[i] for i in self.nl], like=self.gwfn.g_cryst)

        # Defining Function for Simpsons' Integration
        def simpson(f_r: np.ndarray):
            r12 = 1 / 3
            f_times_dr = f_r * r_ab
            # NOTE: Number of radial points specified in UPF File is expected to be odd. Will fail otherwise
            return r12 * np.sum(
                f_times_dr[0:-2:2] + 4 * f_times_dr[1:-1:2] + f_times_dr[2::2]
            )

        # Computing atomic wavefunctions in reciprocal space across a fine mesh of q-points for interpolation
        self.numq = int(np.ceil(np.sqrt(2 * self.ecut) / DEL_Q + 4))
        self.q = np.arange(self.numq, like=self.gwfn.g_cryst) * DEL_Q
        self.chi_q = np.empty(
            (self.numwfc, self.numq), dtype=np.float128, like=self.gwfn.g_cryst
        )
        for iq in range(self.numq):
            q = self.q[iq]
            for ichi in range(self.numwfc):
                l, rchi = self.lchi[ichi], rchi_[ichi]
                sph_jl_qr = spherical_jn(l, q * r)
                # 4 pi / sqrt(cellvol) prefactor will be multipled later
                self.chi_q[ichi, iq] = simpson(rchi * r * sph_jl_qr)

        # Generating mappings between chi and quantum numbers
        self.chi_idxchi = np.asarray(
            [ibeta for ibeta, l in enumerate(self.lchi) for _ in range(-l, l + 1)],
            like=self.gwfn.g_cryst,
        )

        self.chi_l = np.asarray(
            [l for ibeta, l in enumerate(self.lchi) for _ in range(-l, l + 1)],
            like=self.gwfn.g_cryst,
        )

        self.numchi = len(self.chi_l)

    @qtmlogger.time("atwfc:gen_vchi")
    @lru_cache(maxsize=None)
    def gen_chi(self, gkspc: GkSpace, proj_type = 'atomic', nlstr: str | None = None, ) -> tuple[WavefunGType, NDArray, WavefunGType]:
        
        if proj_type == 'atomic':
            if nlstr is None:
                # Setting Up: Computing spherical coordinates for all :math:`\mathbf{G}+\mathbf{k}`
                WavefunG = get_WavefunG(gkspc, 1)
                numgk = gkspc.size_g

                gk_cryst = gkspc.gk_cryst
                gk_x, gk_y, gk_z = gkspc.gk_cart
                gk_norm = gkspc.gk_norm
                chi_fac = FPI / np.sqrt(gkspc.reallat_cellvol)

                where_gk_norm_nonzero = np.where(gk_norm > 1e-7)
                theta = np.zeros_like(gk_z)
                theta[where_gk_norm_nonzero] = np.arccos(
                    gk_z[where_gk_norm_nonzero] / gk_norm[where_gk_norm_nonzero]
                )
                phi = np.arctan2(gk_y, gk_x)

                chi_atom = gkspc.allocate_array((self.numchi, numgk))
                idx_gk = np.rint(gk_norm / DEL_Q).astype("i8")
                xmin0 = gk_norm / DEL_Q - idx_gk
                xmin1 = xmin0 - 1
                xmin2 = xmin0 - 2
                xmin3 = xmin0 - 3

                idxvchi_ = 0
                # Constructing KB Projectors for a single atom
                for idxchi in range(self.numwfc):
                    # Lagrange Interpolation for radial part
                    beta_gk = chi_fac * (
                        self.chi_q[idxchi][idx_gk + 0]
                        * xmin1
                        * xmin2
                        * xmin3
                        / ((0 - 1) * (0 - 2) * (0 - 3))
                        + self.chi_q[idxchi][idx_gk + 1]
                        * xmin0
                        * xmin2
                        * xmin3
                        / ((1 - 0) * (1 - 2) * (1 - 3))
                        + self.chi_q[idxchi][idx_gk + 2]
                        * xmin0
                        * xmin1
                        * xmin3
                        / ((2 - 0) * (2 - 1) * (2 - 3))
                        + self.chi_q[idxchi][idx_gk + 3]
                        * xmin0
                        * xmin1
                        * xmin2
                        / ((3 - 0) * (3 - 1) * (3 - 2))
                    )

                    # Applying angular part using spherical harmonics
                    l = self.lchi[idxchi]
                    for abs_m in range(l + 1):
                        ylm = sph_harm(abs_m, l, phi, theta)
                        if abs_m == 0:
                            chi_atom[abs_m] = ylm * beta_gk
                        else:
                            chi_atom[abs_m] = (
                                -np.sqrt(2) * (-1) ** abs_m * ylm.imag * beta_gk
                            )
                            chi_atom[-abs_m] = (
                                -np.sqrt(2) * (-1) ** abs_m * ylm.real * beta_gk
                            )


                numatoms = self.species.numatoms

                chi_full = WavefunG.empty(numatoms * self.numchi)
                for iat, pos_cryst in enumerate(self.species.r_cryst.T):
                    phase = np.exp(-TPIJ * (pos_cryst @ gk_cryst))
                    chi_iat = chi_full[iat * self.numchi : (iat + 1) * self.numchi]
                    chi_iat.data[:] = phase * chi_atom * (-(1j**self.chi_l)).reshape(-1, 1)

                return chi_full

            else:
                
                WavefunG = get_WavefunG(gkspc, 1)
                numgk = gkspc.size_g

                gk_cryst = gkspc.gk_cryst
                gk_x, gk_y, gk_z = gkspc.gk_cart
                gk_norm = gkspc.gk_norm
                chi_fac = FPI / np.sqrt(gkspc.reallat_cellvol)

                where_gk_norm_nonzero = np.where(gk_norm > 1e-7)
                theta = np.zeros_like(gk_z)
                theta[where_gk_norm_nonzero] = np.arccos(
                    gk_z[where_gk_norm_nonzero] / gk_norm[where_gk_norm_nonzero]
                )
                phi = np.arctan2(gk_y, gk_x)

                idx_gk = np.rint(gk_norm / DEL_Q).astype("i8")
                xmin0 = gk_norm / DEL_Q - idx_gk
                xmin1 = xmin0 - 1
                xmin2 = xmin0 - 2
                xmin3 = xmin0 - 3
                idxchi = self.nl.index(nlstr)
                l = self.lchi[idxchi]
                chi_atom = gkspc.allocate_array((2*l+1, numgk))
                # Lagrange Interpolation for radial part
                beta_gk = chi_fac * (
                    self.chi_q[idxchi][idx_gk + 0]
                    * xmin1
                    * xmin2
                    * xmin3
                    / ((0 - 1) * (0 - 2) * (0 - 3))
                    + self.chi_q[idxchi][idx_gk + 1]
                    * xmin0
                    * xmin2
                    * xmin3
                    / ((1 - 0) * (1 - 2) * (1 - 3))
                    + self.chi_q[idxchi][idx_gk + 2]
                    * xmin0
                    * xmin1
                    * xmin3
                    / ((2 - 0) * (2 - 1) * (2 - 3))
                    + self.chi_q[idxchi][idx_gk + 3]
                    * xmin0
                    * xmin1
                    * xmin2
                    / ((3 - 0) * (3 - 1) * (3 - 2))
                )

                # Applying angular part using spherical harmonics
                for abs_m in range(l + 1):
                    ylm = sph_harm(abs_m, l, phi, theta)
                    if abs_m == 0:
                        chi_atom[0] = ylm * beta_gk
                    else:
                        chi_atom[abs_m] = (
                            -np.sqrt(2) * (-1)  * ylm.real * beta_gk
                        )
                        chi_atom[-abs_m] = (
                            -np.sqrt(2) * (-1)  * ylm.imag * beta_gk
                        )

                # Generating Chi corresponding to all atoms
                chi_arr = []
                numatoms = self.species.numatoms
                chi_full = WavefunG.empty(int(numatoms * (2*l+1)))
                for iat, pos_cryst in enumerate(self.species.r_cryst.T):
                    phase = np.exp(-TPIJ * (pos_cryst @ gk_cryst))
                    chi_iat = chi_full[iat * (2*l+1) : (iat + 1) * (2*l+1)]
                    chi_iat.data[:] = phase * chi_atom * (-(1j**l))
                    chi_iat.normalize()
                    chi_arr.append(chi_iat)
                return chi_arr
        else:
            print("unsupported projection type")

        