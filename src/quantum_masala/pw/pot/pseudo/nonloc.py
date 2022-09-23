import numpy as np
from scipy.special import spherical_jn, sph_harm
from scipy.linalg import block_diag

from quantum_masala.core import AtomBasis, GSpace, GSpaceWfn
from quantum_masala.core.ppdata import UPFv2Data

DEL_Q = 1e-2


class PPDataNonLocal:
    """ """

    def __init__(self, sp: AtomBasis, gspc: GSpace):
        # Setting Up
        self.species = sp
        if not isinstance(self.species.ppdata, UPFv2Data):
            raise NotImplementedError("only 'UPFv2Data' supported")

        self.ecut = gspc.ecut / 4

        cellvol = gspc.reallat_cellvol
        _4pibsqv = 4 * np.pi / np.sqrt(cellvol)

        ppdata = self.species.ppdata
        r = ppdata.r
        r_ab = ppdata.r_ab

        self.numbeta = ppdata.number_of_proj
        self.beta_l = np.array(ppdata.l_kb_l)
        l_kb_rbeta = ppdata.l_kb_rbeta

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
        self.q = np.arange(self.numq) * DEL_Q
        self.beta_q = np.empty((self.numbeta, self.numq), dtype="f8")
        for iq in range(self.numq):
            q = self.q[iq]
            for ibeta in range(self.numbeta):
                l, rbeta = self.beta_l[ibeta], l_kb_rbeta[ibeta]
                sph_jl_qr = spherical_jn(l, q * r)
                self.beta_q[ibeta, iq] = _4pibsqv * simpson(rbeta * r * sph_jl_qr)

        # Generating mappings between KB projectors and quantum numbers
        self.vkb_idxbeta = np.concatenate(
            [[ibeta for _ in range(-l, l + 1)] for ibeta, l in enumerate(self.beta_l)]
        )
        self.vkb_l = np.concatenate(
            [[l for _ in range(-l, l + 1)] for l in self.beta_l]
        )
        self.vkb_m = np.concatenate(
            [
                [((i + 1) // 2) * (-1) ** (i % 2) for i in range(2 * l + 1)]
                for l in self.beta_l
            ]
        )
        self.numvkb = len(self.vkb_l)

        self.dij_beta = ppdata.dij

    def gen_vkb_dij(self, gwfc: GSpaceWfn):
        # Setting Up: Computing spherical coordinates for all :math:`\mathbf{G}+\mathbf{k}`
        numgk = gwfc.numgk
        gk_cryst = gwfc.cryst
        gk_x, gk_y, gk_z = gwfc.cart
        gk_norm = gwfc.norm

        theta = np.arccos(
            np.divide(gk_z, gk_norm, out=np.zeros_like(gk_z), where=gk_norm > 1e-7)
        )
        phi = np.arctan2(gk_y, gk_x)

        l_vkb_atom = np.empty((self.numvkb, numgk), dtype="c16")
        dij_atom = np.zeros((self.numvkb, self.numvkb), dtype="c16")
        idx_gk = np.rint(gk_norm / DEL_Q).astype("i8")
        xmin0 = gk_norm / DEL_Q - idx_gk
        xmin1 = xmin0 - 1
        xmin2 = xmin0 - 2
        xmin3 = xmin0 - 3

        idxvkb = 0
        # Constructing KB Projectors for a single atom
        for idxbeta in range(self.numbeta):
            # Lagrange Interpolation for radial part
            beta_gk = (
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
        l_vkb_full = np.empty((numatoms * self.numvkb, numgk), dtype="c16")
        for iat, pos_cryst in enumerate(self.species.cryst.T):
            phase = np.exp((-2 * np.pi * 1j) * (pos_cryst @ gk_cryst))
            l_vkb_full[iat * self.numvkb : (iat + 1) * self.numvkb] = (
                phase * l_vkb_atom * (-(1j**self.vkb_l)).reshape(-1, 1)
            )
        dij_full = block_diag(*[dij_atom for _ in range(numatoms)])

        return l_vkb_full, dij_full
