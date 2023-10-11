from __future__ import annotations
__all__ = ['compute']
import numpy as np
from scipy.special import erfc

from qtm.crystal import Crystal
from qtm.gspace import GSpace

EWALD_ERR_THR = 1e-7


def compute(crystal: Crystal, gspc: GSpace) -> float:
    reallat, l_species = crystal.reallat, crystal.l_atoms
    latvec, cellvol = reallat.primvec, reallat.cellvol
    _2pibv = 2 * np.pi / cellvol

    l_charges = np.asarray(
        sum(([sp.valence, ] * sp.numatoms for sp in l_species), []),
        like=gspc.g_cryst
    )
    numat = l_charges.shape[0]
    r_cryst_all = np.concatenate([sp.r_cryst for sp in l_species], axis=1)
    ecut = gspc.ecut
    g_cryst = gspc.g_cryst
    g_norm2 = gspc.g_norm2
    struct_fac = np.sum(
        np.exp(-2 * np.pi * 1j * r_cryst_all.T @ g_cryst) * l_charges.reshape(-1, 1),
        axis=0,
    )

    def err_bounds(_alpha):
        return (
            np.sum(l_charges) ** 2
            * np.sqrt(_alpha / np.pi)
            * erfc(np.sqrt(ecut / 2 / _alpha))
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

    rij_cryst = r_cryst_all.reshape((3, -1, 1)) - r_cryst_all.reshape(
        (3, 1, -1)
    )
    qij = l_charges.reshape(-1, 1) * l_charges.reshape(1, -1)
    ni = np.floor(4 / beta / np.linalg.norm(latvec, axis=1)).astype('i8') + 1

    xi = [np.arange(-n, n + 1, dtype='i8', like=g_cryst) for n in ni.tolist()]
    N = np.array(np.meshgrid(*xi, indexing='ij'),
                 like=g_cryst).reshape((3, -1, 1, 1))
    Rij_cryst = N + np.expand_dims(rij_cryst, axis=1)
    Rij_cart = reallat.cryst2cart(Rij_cryst)
    Rij_norm = np.linalg.norm(Rij_cart, axis=0)

    # Finding index where N=0
    i_N0 = ni[0] * (2 * ni[1] + 1) * (2 * ni[2] + 1) \
        + ni[1] * (2 * ni[2] + 1) + ni[2]
    # CUPY_NOTE: np.fill_diagonal is bugged for some reason here
    # np.fill_diagonal(Rij_norm[i_N0], 1E50)
    for iat in range(numat):
        Rij_norm[i_N0, iat, iat] = np.inf
    sij = qij / Rij_norm * erfc(beta * Rij_norm)
    E_S = 0.5 * np.sum(sij)

    f = np.exp(-g_norm2[1:] / (4 * alpha)) / g_norm2[1:]
    E_L = _2pibv * (
        np.sum(f * np.abs(struct_fac[1:]) ** 2) - np.sum(qij) / (4 * alpha)
    )
    return E_S + E_L - E_self
