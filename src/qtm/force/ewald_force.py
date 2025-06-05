from scipy.special import erfc
import gc

import numpy as np

from qtm.lattice import ReciLattice
from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.constants import ELECTRON_RYD, PI, RYDBERG_HART
from qtm.dft import DFTCommMod
EWALD_ERR_THR = 1e-6  # TODO: In quantum masala ewald energy code it is set to 1e-7

'''def rgen2(rmax: float,
         max_num: int,
         beta: float,
         latvec: np.ndarray,
         dtau: np.ndarray):
    """r max: the maximum radius we take into account

    max_num: maximum number of r vectors

    latvec: lattice vectors, each column representing a vector.
            Numpy array with dimensions (3,3)

    recvec: reciprocal lattice vectors, each column representing a vector.
            Numpy array with dimensions (3,3)

    dtau:difference between atomic positions. numpy array with shape (3,)"""
    file=open("rgen2.txt", "a")
    if rmax == 0:
        raise ValueError("rmax is 0, grid is non-existent.")
    # making the grid
    n = np.floor(2 / beta / np.linalg.norm(latvec, axis=1)).astype('i8') + 2
    ni = n[0]
    nj = n[1]
    nk = n[2]
    print("ni, nj, nk" , ni, nj, nk)
    vec_num = 0
    r = np.zeros((3, max_num))
    r_norm = np.zeros((max_num))
    for i in range(-ni, ni):
        for j in range(-nj, nj):
            for k in range(-nk, nk):
                t = i * latvec[:, 0] + j * latvec[:, 1] + k * latvec[:, 2] - dtau
                t_norm = np.linalg.norm(t)
                if t_norm < rmax and np.abs(t_norm) > 1e-5:
                    r[:, vec_num] = t
                    r_norm[vec_num] = t_norm
                    vec_num += 1
                if vec_num >= max_num:
                    raise ValueError(f"maximum allowed value of r vectors are {max_num}, got {vec_num}. ")
    return r, r_norm'''

def transgen(latvec: np.ndarray,
         rmax: float):
    """r max: the maximum radius we take into account

    max_num: maximum number of r vectors

    latvec: lattice vectors, each column representing a vector.
            Numpy array with dimensions (3,3)

    recvec: reciprocal lattice vectors, each column representing a vector.
            Numpy array with dimensions (3,3)

    dtau: difference between atomic positions. numpy array with shape (3,)"""


    # making the grid
    n = np.floor(1/np.linalg.norm(latvec, axis=1)*rmax).astype('i8') + 2
    ni = n[0]
    nj = n[1]
    nk = n[2]
    l0 = latvec[:, 0]
    l1 = latvec[:, 1]
    l2 = latvec[:, 2]
    i=np.arange(-ni, ni)
    j=np.arange(-nj, nj)
    k=np.arange(-nk, nk)
    l0_trans=np.outer(i, l0)
    l1_trans=np.outer(j, l1)
    l2_trans=np.outer(k, l2)
    l0_trans=l0_trans[np.newaxis, :, np.newaxis, np.newaxis, :]
    l1_trans=l1_trans[np.newaxis, np.newaxis, :, np.newaxis, :]
    l2_trans=l2_trans[np.newaxis, np.newaxis, np.newaxis, :, :]
    trans=np.squeeze(l0_trans+l1_trans+l2_trans).reshape(-1,3)
    del i, j, k, l0_trans, l1_trans, l2_trans
    return trans

def rgen(trans:np.ndarray,
         dtau:np.ndarray,
         max_num:float,
            rmax:float
         ):
    if rmax == 0:
        raise ValueError("rmax is 0, grid is non-existent.")
    trans_copy=trans.copy()
    trans_copy-=dtau
    norms=np.linalg.norm(trans_copy, axis=1)
    mask=(norms<rmax) & (norms**2>1e-5)
    r=trans_copy[mask]
    r_norm=norms[mask]
    vec_num=r.shape[0]
    del trans_copy, norms, mask
    if vec_num >= max_num:
        raise ValueError(f"maximum allowed value of r vectors are {max_num}, got {vec_num}. ")
    return r.T, r_norm, vec_num

def force_ewald(
        dftcomm: DFTCommMod,
        crystal: Crystal,
        gspc: GSpace,
        gamma_only: bool = False) -> np.ndarray:
    """This code implements ewald forces given the crystal structure and Gspace.

        Input:
        crystal: The crystal structure of the substance. Type crystal
        gspc: The G space characteristics. Type GSpace

        Note: Quantum Espresso uses alat units. Quantum MASALA uses cryst units.

        Output:
        An array of shape (3, numatom) depicting ewald forces acting on each atom.
        numatom being the number of atoms in the crystal.

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

    del gg_nonzero, gcart_nonzero, str_arg, str_fac, sumnb

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
            del dtau_frac, dtau, recvec
            rgenerate = rgen(trans=trans,
                             dtau=dtau0,
                             max_num=max_num,
                             rmax=rmax
                             )
            r, r_norm, vec_num= rgenerate
            if vec_num!=0:
                rr=r_norm*alat
                fact=2*valence_all[atom1] * valence_all[atom2] * ELECTRON_RYD ** 2 /2* (
                                erfc(eta * rr) / rr + 2 * eta / np.sqrt(PI) * np.exp(-eta ** 2 * rr ** 2)) / rr ** 2
                r_eff_vec = r * alat
                F_S[:, atom1] -= np.sum(fact * r_eff_vec, axis=1)
                del fact, r_eff_vec 
            del r, r_norm, vec_num, dtau0
    Force = F_S + F_L
    Force=Force.T
    #Force = crystal.symm.symmetrize_vec(Force)
    #Force-=np.mean(Force, axis=0)
    del trans, F_S, F_L, gspc, crystal, reallat, l_atoms, valence_all, coords_cart_all, dftcomm
    gc.collect()
    return Force
