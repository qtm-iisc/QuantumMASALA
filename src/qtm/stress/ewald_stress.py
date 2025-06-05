import scipy
import numpy as np

from qtm.lattice import ReciLattice
from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.constants import ELECTRON_RYD, PI, RY_KBAR

EWALD_ERR_THR = 1e-7  # TODO: In quantum masala ewald energy code it is set to 1e-7


def rgen(rmax: float,
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

    if rmax == 0:
        raise ValueError("rmax is 0, grid is non-existent.")
    # making the grid
    n = np.floor(4 / beta / np.linalg.norm(latvec, axis=1)).astype('i8') + 2
    ni = n[0]
    nj = n[1]
    nk = n[2]

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
    return r, r_norm, vec_num


def stress_ewald(
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

    idxsort=gspc.idxsort
    gvec_nonzero = gspc.g_cryst[:, 1:]
    gvec_nonzero=(gvec_nonzero.T[idxsort[1:]-1]).T
    gcart_nonzero = gspc.g_cart[:, 1:]
    gcart_nonzero = (gcart_nonzero.T[idxsort[1:] - 1]).T
    gg_nonzero = np.sum(gcart_nonzero * gcart_nonzero, axis=0)
    gnum=gspc.size_g-1

    # getting the crystal characteristics
    l_atoms = crystal.l_atoms
    reallat = crystal.reallat
    alat = reallat.alat
    omega = reallat.cellvol

    latvec = np.array(reallat.axes_alat)
    recilat = ReciLattice.from_reallat(reallat=reallat)
    valence_all = np.repeat([sp.ppdata.valence for sp in l_atoms], [sp.numatoms for sp in l_atoms])

    # concatenated version of coordinate arrays where ith column represents the coordinate of ith atom.
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])

    # calculating the analogous structure factor:
    str_fac = np.sum(np.exp(1j * coords_cart_all.T @ gcart_nonzero) * valence_all.reshape(-1, 1), axis=0)

    # getting the error bounds TODO: write the error formula
    alpha = 2.8

    def err_bounds(_alpha):
        return (
                2
                * np.sum(valence_all) ** 2
                * np.sqrt(_alpha / np.pi)
                * scipy.special.erfc(np.sqrt(gspc.ecut / 2 / _alpha))
        )

    while err_bounds(alpha) > EWALD_ERR_THR:
        alpha -= 0.1
        if alpha < 0:
            raise ValueError(
                f"'alpha' cannot be set for ewald energy calculation; estimated error too large"
            )
    eta = np.sqrt(alpha)
    fact = 2 if gamma_only else 1
    rho_star2 = (np.abs(str_fac)/omega)**2
    g_quant=np.exp(-gg_nonzero / 4 / alpha) / gg_nonzero
    sewald =rho_star2*g_quant
    sewald*=2*PI*fact*ELECTRON_RYD**2
    g_tensor=2*np.einsum('ij, ik->ijk', gcart_nonzero.T, gcart_nonzero.T)
    g_tensor*=((gg_nonzero/(4*alpha)+1)/gg_nonzero*sewald).reshape(-1,1,1)
    S_L=-np.sum(g_tensor, axis=0)

    total_charge=np.sum(valence_all)
    s_self= PI/(2*alpha)*ELECTRON_RYD**2*(total_charge/omega)**2

    sewald = np.sum(sewald)
    s_self=s_self-sewald
    for idx in range(3):
        S_L[idx,idx]-=s_self

    rmax = 4 / eta / alat
    max_num = 50
    S_S = np.zeros((3, 3))
    for atom1 in range(tot_atom):
        for atom2 in range(tot_atom):
            dtau = (coords_cart_all[:, atom1] - coords_cart_all[:, atom2]) / alat
            rgenerate = rgen(rmax=rmax,
                             max_num=max_num,
                             beta=eta,
                             latvec=latvec,
                             dtau=dtau)
            r, r_norm, vec_num = rgenerate
            rr=r_norm*alat
            r_tensor=np.einsum('ij, ik-> ijk', r.T, r.T)*alat**2
            fact = -2*valence_all[atom1] * valence_all[atom2] /(2*omega) *(
                            scipy.special.erfc(eta * rr) / rr + 2 * eta / np.sqrt(PI) * np.exp(-eta ** 2 * rr ** 2)) / rr ** 2 
            rr_nonzero=np.where(rr>1e-5)
            r_tensor[rr_nonzero]*=fact[rr_nonzero].reshape(-1,1,1)
            S_S-=np.sum(r_tensor, axis=0)
    Stress = S_S + S_L
    return Stress*RY_KBAR











