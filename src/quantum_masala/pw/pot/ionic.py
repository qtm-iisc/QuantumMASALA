import numpy as np
from scipy.special import erfc

from quantum_masala.core import Crystal, Rho, GSpace

from .base import LocalPot
from .pseudo.loc import PPDataLocal

EWALD_ERR_THR = 1e-7


class Ionic(LocalPot):
    def __init__(
        self,
        rho: Rho,
        l_pploc: list[PPDataLocal],
    ):
        super().__init__(rho)
        self._g = sum(pploc.vlocal_g for pploc in l_pploc)
        self._r = self.fft_rho.g2r(self._g)
        self._en = self.rho.integral_rho_f_dv(self._r)
        self.sync()

    @property
    def g(self):
        return self._g

    @property
    def r(self):
        return self._r

    @property
    def en(self):
        return self._en

    def sync(self):
        self._g = self.pwcomm.world_comm.Bcast(self._g)
        self._r = self.fft_rho.g2r(self._g)
        self._en = self.rho.integral_rho_f_dv(self._r)

    def compute(self):
        self._en = self.rho.integral_rho_f_dv(self._r)


def compute_ewald_en(crystal: Crystal, gspc: GSpace):
    realspc, l_species = crystal.reallat, crystal.l_species
    latvec, cellvol = realspc.primvec, realspc.cellvol
    _2pibv = 2 * np.pi / cellvol

    l_charges = np.repeat(
        [sp.ppdata.valence for sp in l_species], [sp.numatoms for sp in l_species]
    )
    l_pos_cryst_all = np.concatenate([sp.cryst for sp in l_species], axis=0)

    g_cryst = gspc.cryst
    g_norm2 = gspc.norm2
    struct_fac = np.sum(
        np.exp(-2 * np.pi * 1j * l_pos_cryst_all.T @ g_cryst) * l_charges.reshape(-1, 1),
        axis=0,
    )

    def err_bounds(_alpha):
        return (
            2
            * np.sum(l_charges) ** 2
            * np.sqrt(_alpha / np.pi)
            * erfc(np.sqrt(gspc.ecut / 2 / _alpha))
        )

    alpha = 2.8
    while err_bounds(alpha) > EWALD_ERR_THR:
        alpha -= 0.1
        if alpha < 0:
            raise ValueError(
                f"'alpha' cannot be set for ewald energy calculation; estimated error too large"
            )
    beta = np.sqrt(alpha)

    E_self = beta / np.sqrt(np.pi) * np.sum(l_charges**2)

    rij_cryst = l_pos_cryst_all.reshape((1, -1, 3)) - l_pos_cryst_all.reshape(
        (-1, 1, 3)
    )
    qij = l_charges.reshape(-1, 1) * l_charges.reshape(1, -1)
    ni = np.ceil(4 / beta / np.linalg.norm(latvec, axis=1))
    ni = [int(n) for n in ni]
    xi = [np.fft.fftfreq(n, 1 / n).astype("i4") for n in ni]
    N = np.transpose([np.ravel(arr) for arr in np.meshgrid(*xi)])
    Rij_cryst = N + np.expand_dims(rij_cryst, axis=2)
    Rij_cart = np.tensordot(Rij_cryst, latvec.T, axes=1)
    Rij_norm = np.linalg.norm(Rij_cart, axis=3)
    np.fill_diagonal(Rij_norm[:, :, 0], np.inf)
    sij = np.expand_dims(qij, axis=2) / Rij_norm * erfc(beta * Rij_norm)
    E_S = 0.5 * np.sum(sij)

    f = np.exp(-g_norm2[1:] / (2 * beta) ** 2) / g_norm2[1:]
    E_L = _2pibv * (
        np.sum(f * np.abs(struct_fac[1:]) ** 2) - np.sum(qij) / (2 * beta) ** 2
    )

    return E_S + E_L - E_self
