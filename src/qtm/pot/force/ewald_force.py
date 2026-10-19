from scipy.special import erfc

import numpy as np

from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.constants import ELECTRON_RYD, PI, RYDBERG_HART
from qtm.dft import DFTCommMod
from qtm.pot.ewald_rvecs import transgen, rgen

EWALD_ERR_THR = 1e-6  # TODO: In quantum masala ewald energy code it is set to 1e-7


def force_ewald(
        dftcomm: DFTCommMod,
        crystal: Crystal,
        gspc: GSpace,
        gamma_only: bool = False) -> np.ndarray:
    r"""This code implements ewald forces given the crystal structure and Gspace.

        Input:
        crystal: The crystal structure of the substance. Type crystal
        gspc: The G space characteristics. Type GSpace

        Note: Quantum Espresso uses alat units. Quantum MASALA uses cryst units.
        The Ewald forces are given as:

        $$-\sum_{\vec{L},n }\sum_{j}'Z_iZ_j\frac{\vec{r}_i-\vec{r}_j+\vec{L}n}{|\vec{r}_i-\vec{r}_j+\vec{L}n|\sqrt{2}\sigma	}H(\frac{|\vec{r}_i-\vec{r}_j+\vec{L}n|}{\sqrt{2}\sigma})+\sum_{\vec{k}\neq 0}\sum_jZ_jZ_i\frac{\vec{k}}{2V\epsilon_0k^2}\exp(-\frac{\sigma^2k^2}{2})\sin(\vec{k}.(\vec{r}_j-\vec{r}_i))$$
        where $H(x)$ is given by:
        $$H(x)=\frac{\partial \text{erfc}(x)}{\partial x}-\frac{\text{erfc}(x)}{x}$$
        The first part is calculated in the real-space and the second term is calculated in the reciprocal space. First, in the code all the necessary information
        is extracted from the crystal. 

        In the subsequent part, a parameter named $\alpha$ is defined which is defined $\alpha=2\sigma^2$ and $\eta=\sqrt{2}\sigma$

        In the later part first the reciprocal part of the force is calculated and then the real space part is calculated. In calculating the real part,
        use of complimentary functions are used. 
        
        Output:
        An array of shape (3, numatom) depicting ewald forces acting on each atom.
        numatom being the number of atoms in the crystal

        Primvec of Gspc.recilat: The columns represent the reciprocal lattice vectors"""
    # getting the characteristic of the g_vectors:
    gcart_nonzero = gspc.g_cart.T[np.linalg.norm(gspc.g_cart.T, axis=1) > 1e-5]
    gg_nonzero = np.linalg.norm(gcart_nonzero, axis=1)**2
    gcart_nonzero = gcart_nonzero.T

    # getting the crystal characteristics
    l_atoms = crystal.l_atoms
    reallat = crystal.reallat
    alat = reallat.alat
    omega = reallat.cellvol

    latvec = np.array(reallat.axes_alat)
    valence_all = np.repeat([sp.ppdata.valence for sp in l_atoms], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])
    str_arg = coords_cart_all.T @ gcart_nonzero

    # calculating the analogous structure factor:
    ##This is actually the conjugate of the structure factor. For optimization reasons 
    #it is calculated directly. Put a -1j in the exponential to get the structure factor.
    str_fac = np.sum(np.exp(1j * str_arg) * valence_all.reshape(-1, 1), axis=0)
    alpha = 2.8

    def err_bounds(_alpha):
        return (
                2
                * np.sum(valence_all) ** 2
                * np.sqrt(_alpha / np.pi)
                * erfc(np.sqrt( gspc.ecut / 2/_alpha))
        )
    
    while err_bounds(alpha) > EWALD_ERR_THR:
        alpha -= 0.1
        if alpha < 0:
            raise ValueError(
                f"'alpha' cannot be set for ewald energy calculation; estimated error too large"
            )
    eta = np.sqrt(alpha)
    fact = 4 if gamma_only else 2
    
    str_fac *= np.exp(-gg_nonzero / 4 / alpha) / gg_nonzero
    sumnb = np.cos(str_arg) * np.imag(str_fac) - np.sin(str_arg) * np.real(str_fac)
    F_L = gcart_nonzero @ sumnb.T
    F_L *= - valence_all.T
    F_L *= 2 * PI / omega
    F_L *= fact/RYDBERG_HART
    ##If the intra-pwgrp size is not 1, then we need to sum the forces over the intra-pwgrp
    if dftcomm.pwgrp_intra!=None: F_L=dftcomm.pwgrp_intra.allreduce(F_L)

    rmax = 5 / eta / alat
    max_num = 100
    trans=transgen(latvec=latvec, rmax=rmax)

    F_S = np.zeros((3, tot_atom))
    for atom1 in range(tot_atom):
        for atom2 in range(tot_atom):
            dtau = (coords_cart_all[:, atom1] - coords_cart_all[:, atom2]) / alat
            recvec=np.array(gspc.recilat.axes_tpiba)
            dtau@=recvec.T
            dtau_frac=dtau-np.round(dtau)
            dtau0=latvec.T@dtau_frac
            r, r_norm, vec_num = rgen(trans=trans,
                             dtau=dtau0,
                             max_num=max_num,
                             rmax=rmax
                             )
            if vec_num!=0:
                rr=r_norm*alat
                fact=2*valence_all[atom1] * valence_all[atom2] * ELECTRON_RYD ** 2 /2* (
                                erfc(eta * rr) / rr + 2 * eta / np.sqrt(PI) * np.exp(-eta ** 2 * rr ** 2)) / rr ** 2
                r_eff_vec = r * alat
                F_S[:, atom1] -= np.sum(fact * r_eff_vec, axis=1)
    Force = (F_S + F_L).T
    return Force
