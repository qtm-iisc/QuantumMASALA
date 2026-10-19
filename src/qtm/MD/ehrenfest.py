from __future__ import annotations

from typing import Optional

import numpy as np
from qtm.config import qtmconfig
from qtm.containers.field import FieldGType, FieldRType
from qtm.crystal.crystal import Crystal, BasisAtoms
from qtm.dft.kswfn import KSWfn
from qtm.pot import hartree, xc
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.pseudo.nloc import NonlocGenerator
from qtm.tddft_gamma.propagate import propagate
from qtm.pot.force import force
from qtm.dft.config import DFTCommMod
from qtm.constants import BOLTZMANN_RYD, M_NUC_RYD, MASS_SI

__all__ = ["scf", "EnergyData", "Iterprinter"]

def Ehrenfest(
    dftcomm: DFTCommMod,
    crystal: Crystal,
    rho_start: FieldGType,
    wfn_gamma: list[list[KSWfn]],
    T_init: float,
    time_step_N: float,
    time_step_e: float,
    numstep: int,
    vel_start:Optional[np.ndarray | None] = None,
    libxc_func: Optional[tuple[str, str]] = None,
):
    # Begin setup ========================================
    ##Setting up the crystal parameters
    l_atoms = crystal.l_atoms
    tot_num = np.sum([sp.numatoms for sp in l_atoms])
    num_in_types=np.array([sp.numatoms for sp in l_atoms])
    ppdat_cryst=np.array([sp.ppdata for sp in l_atoms])
    label_cryst=np.array([sp.label for sp in l_atoms])
    mass_cryst=np.array([sp.mass for sp in l_atoms])*M_NUC_RYD
    mass_all=np.repeat([sp.mass for sp in l_atoms], [sp.numatoms for sp in l_atoms])*M_NUC_RYD
    tot_mass=np.sum(mass_all)
    num_typ = len(l_atoms)
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis =1).T
    reallat=crystal.reallat
    alat=reallat.alat
    coords_ref=coords_cart_all/reallat.alat


    config = qtmconfig

    numel = crystal.numel

    is_spin = len(wfn_gamma[0]) > 1
    is_noncolin = wfn_gamma[0][0].is_noncolin
    numbnd = wfn_gamma[0][0].numbnd

    gspc_rho = rho_start.gspc
    gspc_wfn = gspc_rho
    gkspc = wfn_gamma[0][0].gkspc

    ##INIT-MD
    #region: Initializing the Molecular Dynamics simulations
    mass_si=mass_all*MASS_SI
    ##First we assign velocities to the atoms
    vel=np.random.rand(tot_num, 3)-0.5
    ##Calculate the momentum
    momentum=mass_all*vel.T

    momentum_T=momentum.T
    momentum_T-=np.mean(momentum, axis=1)
    ##Calculate the new velocity after subtracting the momentum of the center of mass
    
    vel=momentum_T.T/mass_all

    ##Calculate the kinetic energy
    ke=0.5*np.sum(mass_all*(vel)**2)
    ##Calculate the temperature
    T=2*ke/(3*tot_num*BOLTZMANN_RYD)
    ##Rescale the velocities to the desired temperature
    vel*=np.sqrt(T_init/T)
    if type(vel_start)==np.ndarray:
        vel=vel_start.T

    time_array=[]
    energy_array=[]
    temperature_array=[]

    def force_wrapper(dftcomm:DFTCommMod | None,
                      wfn_gamma:list[list[KSWfn]],
                      crystal:Crystal,
                      rho_start:FieldGType,
                      gamma_only:bool=False,
                      verbosity:bool=False,
                      libxc_func:Optional[tuple[str, str]] = None,
                      ):
        ##Create the reciprocal space from rho
        numbnd=wfn_gamma[0][0].numbnd
        gspc_rho = rho_start.gspc
        gspc_wfn = gspc_rho
        gkspc = wfn_gamma[0][0].gkspc

        v_ion = rho_start.zeros(1)
        rho_core = rho_start.zeros(1)

        l_nloc = []
        for sp in crystal.l_atoms:
            v_ion_typ, rho_core_typ, v_ion_nomult, rho_ion_nomult= loc_generate_pot_rhocore(sp, gspc_rho)
            v_ion += v_ion_nomult
            rho_core += rho_core_typ
            n_loc_gen=NonlocGenerator(sp, gspc_wfn)
            vkb_full, dij_full, vkb_diag=n_loc_gen.gen_vkb_dij(gkspc)
            l_nloc.append((vkb_full, dij_full))
        l_nloc=[l_nloc]
        v_ion= v_ion.to_r()
        rho_out: FieldGType
        v_hart: FieldRType
        v_xc: FieldRType
        v_loc: FieldRType

        if libxc_func is None:
            libxc_func = xc.get_libxc_func(crystal)
        else:
            xc.check_libxc_func(libxc_func)

        def compute_v_local(rho_):
            nonlocal v_hart, v_xc, v_loc
            nonlocal rho_core
            v_hart, en_hartree = hartree.compute(rho_)
            v_xc, en_xc, GGA = xc.compute(rho_, rho_core, *libxc_func)
            v_loc = v_ion + v_hart + v_xc
            v_loc *= 1 / np.prod(gspc_wfn.grid_shape)
            return v_loc
        
        v_loc=compute_v_local(rho_start)
        v_loc=v_loc.to_g()
        
        force_calc=force(dftcomm=dftcomm,
                        numbnd=numbnd,
                        wavefun=wfn_gamma, 
                        crystal=crystal,
                        gspc=gspc_wfn,
                        rho=rho_start,
                        vloc=v_loc,
                        nloc_dij_vkb=l_nloc,
                        gamma_only=gamma_only,
                        verbosity=verbosity)[0]
        return force_calc
    
    def dipole_updater(
        step: int,
        rho: FieldGType
    ) -> None:
        """Update the dipole moment at each step."""
        # This function is a placeholder and should be implemented as needed.
        pass


    ## Starting of the loops
    ##Making the Hamiltonian Propagator  from the crystal:
    sp=crystal.l_atoms[0]
    for istep in range(numstep):
        
        ##First propagate the nuclear coordinates
        ##Half Step
        force_0=force_wrapper(dftcomm=dftcomm,
                        wfn_gamma=wfn_gamma, 
                        crystal=crystal,
                        rho_start=rho_start,
                        gamma_only=False,
                        verbosity=False,
                        libxc_func=libxc_func)
        
        acc_0=(force_0.T/mass_all)
        coords_cart_all+=vel.T*time_step_N+0.5*acc_0.T*time_step_N**2
        vel+=0.5*acc_0*time_step_N

        ###THis code only works for homogeneous systems.
        #Construct the updated crystal
        #Extract the properties of the basis atoms
        counter=0
        updated_sp=[]
        for sp in crystal.l_atoms:
            label=sp.label
            ppdata=sp.ppdata
            mass=sp.mass
            coords_cart_tmp=(coords_cart_all[counter:counter+sp.numatoms])/alat
            new_sp=BasisAtoms.from_alat(label,
                                        ppdata,
                                        mass,
                                        reallat,
                                        coords_cart_tmp)
            updated_sp.append(new_sp)
            counter+=sp.numatoms

        crystal=Crystal(reallat, updated_sp)

        force_1by4=force_wrapper(dftcomm=dftcomm,
                        wfn_gamma=wfn_gamma, 
                        crystal=crystal,
                        rho_start=rho_start,
                        gamma_only=False,
                        verbosity=False,
                        libxc_func=libxc_func)
        acc_1by4=(force_1by4.T/mass_all)
        vel+=0.5*acc_1by4*time_step_N

        counter=0
        updated_sp=[]
        for sp in crystal.l_atoms:
            label=sp.label
            ppdata=sp.ppdata
            mass=sp.mass
            coords_cart_tmp=(coords_cart_all[counter:counter+sp.numatoms])/alat
            new_sp=BasisAtoms.from_alat(label,
                                        ppdata,
                                        mass,
                                        reallat,
                                        coords_cart_tmp)
            updated_sp.append(new_sp)
            counter+=sp.numatoms
        crystal=Crystal(reallat, updated_sp)

        ##Now we propagate the electronic wavefunction
        ##Suzuki Trotter method
        p_1=1/(4-4**(1/3))
        p_2=p_1
        p_3=1-4*p_1
        p_4=p_1
        p_5=p_1
        p=np.array([p_1, p_2, p_3, p_4, p_5/2])
        Time=np.cumsum(p)*time_step_e

        for i in range (5):
            propagate(comm_world=dftcomm.image_comm,
                    crystal=crystal,
                    rho_start=rho_start,
                    wfn_gamma=wfn_gamma,
                    time_step=Time[i],
                    numstep=1,
                    dipole_updater=dipole_updater,
                    libxc_func=libxc_func)
        
        rho_start = (wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g())

        ##Now again we update the nuclear coordinates
        force_1by2=force_wrapper(dftcomm=dftcomm,
                        wfn_gamma=wfn_gamma, 
                        crystal=crystal,
                        rho_start=rho_start,
                        gamma_only=False,
                        verbosity=False,
                        libxc_func=libxc_func)
        
        acc_1by2=(force_1by2.T/mass_all)
        coords_cart_all+=vel.T*time_step_N+0.5*acc_1by2.T*time_step_N**2
        vel+=0.5*acc_0*time_step_N

        ###THis code only works for homogeneous systems.
        counter=0
        updated_sp=[]
        for sp in crystal.l_atoms:
            label=sp.label
            ppdata=sp.ppdata
            mass=sp.mass
            coords_cart_tmp=(coords_cart_all[counter:counter+sp.numatoms])/alat
            new_sp=BasisAtoms.from_alat(label,
                                        ppdata,
                                        mass,
                                        reallat,
                                        coords_cart_tmp)
            updated_sp.append(new_sp)
            counter+=sp.numatoms
        crystal=Crystal(reallat, updated_sp)

        force_3by4=force_wrapper(dftcomm=dftcomm,
                        wfn_gamma=wfn_gamma, 
                        crystal=crystal,
                        rho_start=rho_start,
                        gamma_only=False,
                        verbosity=False,
                        libxc_func=libxc_func)
        acc_3by4=(force_3by4.T/mass_all)
        vel+=0.5*acc_3by4*time_step_N
        counter=0
        updated_sp=[]
        for sp in crystal.l_atoms:
            label=sp.label
            ppdata=sp.ppdata
            mass=sp.mass
            coords_cart_tmp=(coords_cart_all[counter:counter+sp.numatoms])/alat
            new_sp=BasisAtoms.from_alat(label,
                                        ppdata,
                                        mass,
                                        reallat,
                                        coords_cart_tmp)
            updated_sp.append(new_sp)
            counter+=sp.numatoms
        crystal=Crystal(reallat, updated_sp)

    return coords_cart_all


