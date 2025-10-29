from __future__ import annotations

import numpy as np

from typing import Literal
from qtm.crystal import Crystal, CrystalSymm
from qtm.pseudo.atomic_wfc import AtwfcGenerator
from qtm.gspace import GSpace, GkSpace
from qtm.constants import RYDBERG
from qtm.constants import FPI, PI, TPI
from qtm.symm import d_matrix
from qtm.symm.d_matrix import real_sph_harm, compute_spherical_harmonics as sph_harm

__all__ = ["dftU"]

def compute_spherical_harmonics(rl, maxl=3):
    """
    Given an array of points rl (shape: num_points x 3), compute the spherical
    harmonics for l = 0, 1, 2, 3. The output array Y has shape (num_points, 16)
    with the ordering:
      - Column 0: l=0, m=0
      - Columns 1-3: l=1, m=0,1,-1
      - Columns 4-8: l=2, m=0,1,-1,2,-2
      - Columns 9-15: l=3, m=0,1,-1,2,-2,3,-3
    """
    num_points = rl.shape[1]
    Y = np.zeros((num_points, (maxl+1)**2))
    x = rl[0]
    y = rl[1]
    z = rl[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    r[r == 0] = 1e-12
    theta = np.arccos(z / r)  # polar angle [0, pi]
    phi = np.arctan2(y, x)    # azimuthal angle [-pi, pi]
    
    index = 0
    for l in range(0, maxl+1):  # l = 0, 1, 2, 3
        Y[:, index] = real_sph_harm(l, 0, theta, phi)
        index += 1
        for m in range(1, l+1):
            Y[:, index] = real_sph_harm(l, m, theta, phi)
            index += 1
            Y[:, index] = real_sph_harm(l, -m, theta, phi)
            index+=1
    return Y

class dftU:
    def __init__(self, crystal: Crystal, nspin, gwfn: GSpace, beta = 0.5, symm_ns=True):
        self.symm_ns = symm_ns
        self.beta = beta
        self.crystal = crystal
        self.crystalSymm = crystal.symm
        self.equiv_atoms = self.crystalSymm.equiv_atoms
        self.l_atoms = crystal.l_atoms
        self.nspin = nspin
        self.gwfn = gwfn
        self.nlchi = self.init_nlchi()
        self.ns = self.init_ns()
        self.hubbard_matrix = {}
        np.set_printoptions(precision=6, suppress=True, threshold=np.inf, linewidth=np.inf)


        for sp in self.l_atoms:
            if sp.is_hubbard:
                if sp.dftU_type[0]=='l':
                    self.hubbard_matrix[sp] = self.hubbard_matrix_(sp, 0)
        self.v_hub, self.eth = self.V_hub()
        self.d_matrices = self.compute_d_matrices()

    def init_nlchi(self):
        nlchi={}
        for sp in self.l_atoms:
            if sp.is_hubbard:
                if sp.ppdata.number_of_wfc>0:
                    nlchi[sp]=AtwfcGenerator(sp, self.gwfn)
                else:
                    raise RuntimeError("Hubbard U is requested but no atomic wavefunctions are provided in the UPF file")
        return nlchi

    def init_ns(self):
        """this routine finds starting ns values given the input"""
        ns = {}
        nspin = self.nspin
        for sp in self.l_atoms:
            if sp.is_hubbard:
                ns[sp] = np.zeros((2*sp.Hubbard_l[0]+1, 2*sp.Hubbard_l[0]+1, nspin, sp.numatoms))
                for atom in range(sp.numatoms):
                    ldim = 2*sp.Hubbard_l[0] + 1
                    non_magnetic = True
                    totoc = sp.Hubbard_occ[0]
                    if nspin == 2:
                        if(sp.start_mag>0):
                            non_magnetic = False
                            majs = 0
                            mins = 1
                        elif (sp.start_mag<0):
                            non_magnetic = False
                            majs = 1
                            mins = 0
                    if not non_magnetic:
                        #atom is magnetic
                        if (totoc>ldim):
                            for m1 in range(ldim):
                                ns[sp][m1,m1,majs,atom] = 1
                                ns[sp][m1,m1,mins,atom] = (totoc-ldim) / ldim
                        else:
                            for m1 in range(ldim):
                                ns[sp][m1,m1,majs,atom] = totoc / ldim

                    else:
                        #atom is non-magnetic
                        for iss in range(nspin):
                            for m1 in range(ldim):
                                ns[sp][m1, m1, iss, atom] = totoc /2 / ldim
                    """
                    for spin in range(nspin):
                        ns[sp][:, :, spin, atom]= np.random.normal(0, 0.01, ns[sp].shape[:2])
                        ns[sp][:, :, spin, atom] = (ns[sp][:, :, spin, atom] + ns[sp][:, :, spin, atom].T) / 2"""
        return ns  
      
    def compute_d_matrices(self):
        s = self.crystalSymm.reallat_rot
        sb = [sisym @ self.crystal.recilat.recvec.T for sisym in s]
        sr = np.array([self.crystal.reallat.latvec @ sbisym for sbisym in sb])/TPI
        d1, d2, d3 = d_matrix.compute_d_matrices(sr)
        for i, sym in enumerate(sr):
            print(f"Symmetry {i}:")
            print(sym)
            print(s[i])
            print("d2matrix:")
            print(d2[i])
        return [None, d1, d2, d3]

    def update_ns(self, l_kswfn_kgrp):
        nsnew={}
        for sp in self.l_atoms:
            if sp.is_hubbard:
                ns_= np.zeros_like(self.ns[sp])
                nsnew[sp] = ns_
                for atom in range(sp.numatoms):
                    for kswfn_k in l_kswfn_kgrp:
                        for spin in range(self.nspin):
                            kswfn_=kswfn_k[spin]
                            chi = self.nlchi[sp].gen_chi(kswfn_k[0].gkspc, sp.proj_type[0], sp.Hubbard_nl[0])[atom]
                            for m1 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                                proj1 = chi[m1].vdot(kswfn_.evc_gk)
                                ns_[m1, m1, spin, atom] += np.real(np.sum(kswfn_.k_weight * kswfn_.occ * np.conjugate(proj1)*proj1))
                                for m2 in range(-sp.Hubbard_l[0], m1):
                                    proj2 = chi[m2].vdot(kswfn_.evc_gk)
                                    ns = np.real(np.sum(kswfn_.k_weight * kswfn_.occ * np.conjugate(proj1)*proj2))
                                    ns_[m1, m2, spin, atom] += ns
                                    ns_[m2, m1, spin, atom] += ns
        if self.symm_ns:      
            self.symmetrise_ns(nsnew)
        self.hermitianise_ns(nsnew)
        beta = self.beta
        for sp in self.l_atoms:
            if sp.is_hubbard:
                self.ns[sp]=(1-beta)*self.ns[sp]+beta*nsnew[sp]
        
        temp=[]
        for sp in self.l_atoms:
            if sp.is_hubbard:
                for atom in range(sp.numatoms):
                    temp.append(np.trace(self.ns[sp][:,:,0,atom]))
                    print(f"Hubbard manifolds occupied spin 0: {temp[-1]}")
                    temp.append(np.trace(self.ns[sp][:,:,1,atom]))
                    print(f"Hubbard manifolds occupied spin 1: {temp[-1]}")
                    
        print(f"Total Hubbard Occupation: {sum(temp)}")
        self.v_hub, self.eth = self.V_hub()
        return
    
    def symmetrise_ns(self, nss):
        for sp in self.l_atoms:
            if sp.is_hubbard:
                equiv_atoms = self.equiv_atoms[sp]
                ns_ = nss[sp]
                ns = np.zeros_like(ns_)
                d = self.d_matrices[sp.Hubbard_l[0]] 
                nsym = d.shape[0]
                for atom in range(sp.numatoms):
                    for spin in range(self.nspin):
                        for m1 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                            for m2 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                                for isym in range(nsym):
                                    nb = equiv_atoms[isym, atom]
                                    for m0 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                                        for m00 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                                            ns[m1, m2, spin, atom] += d[isym, m0, m1] * ns_[m0, m00, spin, nb] * d[isym, m00, m2]/nsym
                ns_[:,:,:,:] = ns
        return

    def hermitianise_ns(self, nss):
        for sp in self.l_atoms:
            if sp.is_hubbard:
                ns_ = nss[sp]
                for atom in range(sp.numatoms):
                    for spin in range(self.nspin):
                        for m1 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                            for m2 in range(m1, sp.Hubbard_l[0]+1):
                                ns_[m1, m2, spin, atom] = (ns_[m2, m1, spin, atom]+ns_[m1, m2, spin, atom])/2
                                ns_[m2, m1, spin, atom] = ns_[m1, m2, spin, atom]
        return

    def delta_e(self):
        e = 0
        for sp in self.l_atoms:
            if sp.is_hubbard:
                if sp.dftU_type[0]=='d' or sp.dftU_type[0]=='l':
                    e+= -np.sum(self.ns[sp]*self.v_hub[sp])
        return e if self.nspin==2 else 2*e


    def print_Vhub(self, sp, atom, vhub):
        print(f"Atom: {sp.label}, Number: {atom}")
        if self.nspin == 2:
            print("Spin 1")
            print("V_hub")
            print(vhub[sp][:,:,1,atom])
            print("ns")
            print(self.ns[sp][:,:,1,atom])
        print("Spin 0")
        print("V_hub")
        print(vhub[sp][:,:,0,atom])
        print("ns")
        print(self.ns[sp][:,:,0,atom])

    def V_hub(self):
        eth = 0
        eth_dc = 0.0
        eth_u = 0.0
        v_hub = {}
        for sp in self.l_atoms:
            if sp.is_hubbard:
                if sp.dftU_type[0]=='d':
                    ns_= self.ns[sp]
                    v_hub_ = np.zeros_like(ns_)
                    v_hub[sp] = v_hub_
                    effU = sp.Hubbard_U[0] - sp.Hubbard_J[0]
                    Hubbard_alpha = 0
                    for atom in range(sp.numatoms):
                        for spin in range(self.nspin):
                            for m1 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                                eth += (Hubbard_alpha+0.5*effU)*ns_[m1,m1,spin,atom]
                                v_hub_[m1, m1, spin, atom]+= Hubbard_alpha + 0.5*effU
                                for m2 in range(-sp.Hubbard_l[0], sp.Hubbard_l[0]+1):
                                    eth -= 0.5*effU*ns_[m2,m1,spin,atom]*ns_[m1, m2, spin, atom]
                                    v_hub_[m1, m2, spin, atom]-=effU*ns_[m2, m1, spin, atom]
                        self.print_Vhub(sp, atom, v_hub)
                elif sp.dftU_type[0]=='l':

                    ns_ = self.ns[sp]
                    v_hub_ = np.zeros_like(ns_)
                    v_hub[sp] = v_hub_
                    # Allocate (build) the u_matrix using a helper function.
                    # u_matrix should be a 4-index array with dimensions
                    # (2*Hubbard_lmax+1, 2*Hubbard_lmax+1, 2*Hubbard_lmax+1, 2*Hubbard_lmax+1)
                    u_matrix = self.hubbard_matrix[sp]
                    l = sp.Hubbard_l[0]                    
                    # Loop over atoms for this species
                    for atom in range(sp.numatoms):
                        # Calculate total occupation n_tot
                        n_tot = 0.0
                        for spin in range(self.nspin):
                            for m in range(-l, l + 1):
                                n_tot += ns_[m, m, spin, atom]
                        if self.nspin == 1:
                            n_tot *= 2.0

                        # Calculate magnetic moment squared mag2 (only for two spins)
                        mag2 = 0.0
                        if self.nspin == 2:
                            for m in range(-l, l + 1):
                                mag2 += ns_[m, m, 0, atom] - ns_[m, m, 1, atom]
                        mag2 = mag2**2

                        # Compute double-counting (DC) energy term
                        eth_dc += 0.5 * (sp.Hubbard_U[0] * n_tot * (n_tot - 1.0)
                                        - sp.Hubbard_J[0] * n_tot * (0.5 * n_tot - 1.0)
                                        - 0.5 * sp.Hubbard_J[0] * mag2)

                        # Loop over spin channels
                        for spin in range(self.nspin):
                            # Compute occupation for this spin channel
                            other_spin = (self.nspin-spin+1)%2
                            n_spin = 0.0
                            for m in range(-l, l + 1):
                                n_spin += ns_[m, m, spin, atom]

                            # Loop over matrix indices for the potential
                            for m1 in range(-l, l + 1):
                                # Add DC contribution to the diagonal element
                                v_hub_[m1, m1, spin, atom] += (sp.Hubbard_J[0] * n_spin
                                                            + 0.5 * (sp.Hubbard_U[0] - sp.Hubbard_J[0])
                                                            - sp.Hubbard_U[0] * n_tot)
                                for m2 in range(-l, l + 1):
                                    # Loop over the additional indices m3 and m4
                                    for m3 in range(-l, l + 1):
                                        for m4 in range(-l, l + 1):
                                            # Sum over the other spin channel(s) for the first contribution.
                                            # Note: In Fortran, the factor is (MOD(nspin,2)+1):
                                            #   if nspin==2: factor = 1, if nspin==1: factor = 2.
                                            factor = (self.nspin % 2) + 1
                                            for spin1 in range(self.nspin):
                                                v_hub_[m1, m2, spin, atom] += factor* u_matrix[m1, m3, m2, m4] * ns_[m3, m4, spin1, atom]

                                            v_hub_[m1, m2, spin, atom]-=u_matrix[m1, m3, m4, m2]*ns_[m3, m4, spin, atom]
                                            eth_u +=0.5*((u_matrix[m1,m2,m3,m4]-u_matrix[m1,m2, m4, m3])*ns_[m1, m3, spin, atom]*ns_[m2, m4, spin, atom] + u_matrix[m1,m2, m3, m4]*ns_[m1, m3, spin, atom]*ns_[m2, m4, other_spin, atom])

                        self.print_Vhub(sp, atom, v_hub)
        sp = self.l_atoms[0]
        if sp.dftU_type[0]=='l': 
            if self.nspin ==1:
                eth_u = 2*eth_u
            eth = eth_u - eth_dc
            
        print(f"E_hubbard: {eth/ RYDBERG} Ry")  
        return v_hub, eth

    def E_U(self):
        return self.eth

    def V_U(self, l_psi, l_hpsi, ispin, gkspc: GkSpace):
        for sp in self.l_atoms:
            if sp.is_hubbard:
                v_hub_ = self.v_hub[sp]
                chi = self.nlchi[sp].gen_chi(gkspc, sp.proj_type[0], sp.Hubbard_nl[0])
                for iatom, ichi in zip(range(sp.numatoms), chi):
                    v_aux = v_hub_[:,:,ispin, iatom]
                    proj = ichi.vdot(l_psi)
                    proj = v_aux @ proj
                    l_hpsi.zgemm(ichi.data.T, proj.T, 0, 1, 1.0, l_hpsi.data.T, 1.0)

    def hubbard_matrix_(self, sp, index):
        L = sp.Hubbard_l[index]         # actual l value
        U = sp.Hubbard_U[index]         # Hubbard U parameter
        J = sp.Hubbard_J[index]         # Hubbard J parameters (expects at least one element)
        B = sp.Hubbard_B[index]         # Hubbard B parameters (expects at least one element)
        E2 = sp.Hubbard_E2[index]       # Hubbard E2 parameters (expects at least one element)
        E3 = sp.Hubbard_E3[index]       # Hubbard E3 parameters (expects at least one element)
        # Determine number of spherical harmonics

        # Allocate and set up the F coefficients (stored in an array of length 7)
        F = np.zeros(7)
        if L == 0:
            F[0] = U
        elif L == 1:
            F[0] = U
            F[2] = 5.0 * J
        elif L == 2:
            F[0] = U
            F[2] = 5.0 * J + 31.5 * B
            F[4] = 9.0 * J - 31.5 * B
        elif L == 3:
            F[0] = U
            F[2] = (225.0 / 54.0) * J + (32175.0 / 42.0) * E2 + (2475.0 / 42.0) * E3
            F[4] = 11.0 * J - (141570.0 / 77.0) * E2 + (4356.0 / 77.0) * E3
            F[6] = (7361.64 / 594.0) * J + (36808.2 / 66.0) * E2 - 111.54 * E3
        else:
            raise ValueError("lda_plus_u is not implemented for L > 3 ...")

        u_matrix = np.zeros((2 * L + 1, 2 * L + 1, 2 * L + 1, 2 * L + 1))
        
        n2l = (2*L+1)**2
        nl = (L+1)**2
        # Generate 16 random vectors (components in [-0.5, 0.5])
        ap = np.zeros((n2l, nl, nl))
        rl = np.random.rand(3, n2l) - 0.5  # shape: (16, 3)
        
        # Compute spherical harmonics for original points.
        ylm = compute_spherical_harmonics(rl, 2*L) # shape: (16, 16)
        mly = np.linalg.inv(ylm)

        for li in range(nl):
            for lj in range(nl):
                for l in range(n2l):
                    ap[l, li, lj]=0
                    for ir in range(n2l):
                        ap[l, li, lj] += mly[l, ir]*ylm[ir, li]*ylm[ir, lj]
        for m1 in range(0, 2*L + 1):
            for m2 in range(0, 2*L+1):
                for m3 in range(0, 2*L+1):
                    for m4 in range(0, 2*L+1):
                        i=0
                        for k in range(0, 2*L+2, 2):
                            ak = 0
                            for q in range(2*k+1):
                                ak+=ap[i, L**2+m1, L**2+m3]*ap[i, L**2+m2, L**2+m4]
                                i+=1
                            ak = ak*FPI/(2*k+1)
                            u_matrix[m1, m2, m3, m4]+= ak*F[k]
                            i+= 2*(k+1) + 1
        u_matrix_new = np.zeros((2*L+1, 2*L+1, 2*L+1, 2*L+1))
        map = [0,1,-1,2,-2,3,-3]


        for m1 in range(0,2*L+1):
            for m2 in range(0, 2*L+1):
                for m3 in range(0, 2*L+1):
                    for m4 in range(0, 2*L+1):
                        m1_, m2_, m3_, m4_ = map[m1], map[m2], map[m3], map[m4]
                        u_matrix_new[m1_, m2_, m3_, m4_] = u_matrix[m1, m2, m3, m4]
        return u_matrix_new

