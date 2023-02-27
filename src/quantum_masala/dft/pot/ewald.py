__all__ = ['ewald_compute']
import numpy as np
from scipy.special import erfc

from quantum_masala.core import Crystal, GSpace

EWALD_ERR_THR = 1e-7


def ewald_compute(crystal: Crystal, gspc: GSpace) -> float:
    reallat, l_species = crystal.reallat, crystal.l_atoms
    latvec, cellvol = reallat.primvec, reallat.cellvol
    _2pibv = 2 * np.pi / cellvol

    l_charges = np.repeat(
        [sp.ppdata.valence for sp in l_species], [sp.numatoms for sp in l_species]
    )
    l_pos_cryst_all = np.concatenate([sp.cryst for sp in l_species], axis=1)
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

    rij_cryst = l_pos_cryst_all.reshape((3, -1, 1)) - l_pos_cryst_all.reshape(
        (3, 1, -1)
    )
    qij = l_charges.reshape(-1, 1) * l_charges.reshape(1, -1)
    ni = np.floor(4 / beta / np.linalg.norm(latvec, axis=1)).astype('i8') + 2

    xi = [np.fft.fftfreq((2*n + 1), 1/(2*n + 1)).astype("i4")
          for n in ni]
    N = np.array(np.meshgrid(*xi, indexing='ij')).reshape((3, -1, 1, 1))
    Rij_cryst = N + np.expand_dims(rij_cryst, axis=1)
    Rij_cart = reallat.cryst2cart(Rij_cryst)
    Rij_norm = np.linalg.norm(Rij_cart, axis=0)
    np.fill_diagonal(Rij_norm[0, :, :], np.inf)
    sij = qij / Rij_norm * erfc(beta * Rij_norm)
    E_S = 0.5 * np.sum(sij)

    f = np.exp(-g_norm2[1:] / (4 * alpha)) / g_norm2[1:]
    E_L = _2pibv * (
        np.sum(f * np.abs(struct_fac[1:]) ** 2) - np.sum(qij) / (4 * alpha)
    )

    return E_S + E_L - E_self
