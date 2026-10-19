from __future__ import annotations
import gc

from qtm.constants import RYDBERG, BOLTZMANN_RYD, M_NUC_RYD
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, scf
from qtm.MD.rdist import RDist

from qtm import qtmconfig
qtmconfig.fft_backend = 'mkl_fft'

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
__all__ = ['scf', 'EnergyData', 'IterPrinter']

import numpy as np

from qtm.kpts import KList
from qtm.gspace import GSpace
from qtm.containers import FieldGType, get_FieldG

from qtm.dft import DFTCommMod, DFTConfig, KSWfn

from qtm.pot.force import force
from qtm.MD.common import EnergyData, IterPrinter, WfnInit, comm_world

## Max_t is the maximum time that can be elapsed, dt is the time steps, and the T_init is the initial temperature of the system.
##If store_var is set to true then, the variables like energy and temperature are stored and 
# these can be plotted with respect to time if is_plot is set to true

def NVE_MD(dftcomm: DFTCommMod,
          crystal: Crystal,
          max_t: float,
          dt: float,
          T_init: float,
          vel_init: None | np.ndarray,
          kpts: KList,
          grho: GSpace,
          gwfn: GSpace,
          ecut_wfn:float,
          numbnd:int,
          is_spin:bool,
          is_noncolin:bool,
          coords_prev: None | np.ndarray =None,
          symm_rho:bool=True,
          rho_start: FieldGType | tuple[float, ...] | None=None,
          wfn_init: WfnInit | None = None,
          libxc_func: tuple[str, str] | None = None,
          occ_typ: Literal['fixed', 'smear'] = 'smear',
          smear_typ: Literal['gauss', 'fd', 'mv'] = 'gauss',
          e_temp: float = 1E-3,
          conv_thr: float = 1E-6*RYDBERG, 
          maxiter: int = 100,
          diago_thr_init: float = 1E-2*RYDBERG,
          iter_printer: IterPrinter | None = None,
          mix_beta: float = 0.7, mix_dim: int = 8,
          dftconfig: DFTConfig | None = None,
          ret_vxc:bool=False,
          gamma_only:bool=False
          ):
    
    with dftcomm.image_comm as comm:
        #region: Extract the crystal properties
        l_atoms = crystal.l_atoms
        tot_num = np.sum([sp.numatoms for sp in l_atoms])
        num_in_types=np.array([sp.numatoms for sp in l_atoms])
        ppdat_cryst=np.array([sp.ppdata for sp in l_atoms])
        label_cryst=np.array([sp.label for sp in l_atoms])
        mass_cryst=np.array([sp.mass for sp in l_atoms])*M_NUC_RYD
        mass_all=np.repeat([sp.mass for sp in l_atoms], [sp.numatoms for sp in l_atoms])*M_NUC_RYD
        num_typ = len(l_atoms)
        reallat=crystal.reallat
        coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis =1).T
        coords_ref=coords_cart_all/reallat.alat
        ##This is a numatom times 3 array containing the coordinates of all the atoms in the crystal
        ##Extracting the k grid properties

        #endregion: Extract the crystal properties

        ##INIT-MD
        #region: Initializing the Molecular Dynamics simulations
        ##First we assign velocities to the atoms
        vel=np.zeros((tot_num, 3))
        np.random.seed(None)
        if type(vel_init)==None:
            vel=np.random.rand(tot_num, 3)-0.5
            vel=comm.allreduce(vel)  
            vel/=comm.size              ##mean of the velocities across the processors
            momentum=mass_all*vel.T
            momentum_T=momentum.T
            momentum_T-=np.mean(momentum, axis=1)     ##momentum of COM is 0
            vel=momentum_T.T/mass_all
            ke=0.5*np.sum(mass_all*(vel)**2)          ##kinetic energy and temperature are calculated
            del momentum, momentum_T
            T=2*ke/(3*tot_num*BOLTZMANN_RYD)           
            vel*=np.sqrt(T_init/T)     ##velocity is rescaled to match the desired temperature.
        else:
            vel=vel_init.T
        vel_ref=vel.T
        norm_factor=np.mean(np.sum(vel**2, axis=1))     ##Calcualtion of VACF pre requisites
        ke_scale=0.5*np.sum(mass_all*(vel**2))          ##kinetic energy after initializing the velocity
        T_scale=2*ke_scale/(3*tot_num*BOLTZMANN_RYD) 
        if type(coords_prev) is np.ndarray:
            coords_cart_prev=coords_prev   ## temperature after initializing the velocity
        else:
            coords_cart_prev=coords_cart_all-vel.T*dt  
           ## THis is required to initialize the Verlet algorithm
        if dftcomm.image_comm.rank==0: print("the temperature after rescaling is", T_scale, "K")
        del vel

        time_step=int(max_t/dt)
        time_array = np.empty(time_step)
        energy_array = np.empty(time_step)
        temperature_array = np.empty(time_step)
        msd_array= np.empty(time_step)
        vacf_array= np.empty(time_step)
        

        ##Defining the radial distributiob function
        bins=1000
        RDF_array=[]
        rmax=np.array([0.5, 0.5, 0.5])
        r_in, RDF_in=RDist(crystal, coords_cart_all, num_bins=bins, rmax=rmax)
        RDF_array.append(RDF_in)

        #endregion: Initializing the Molecular Dynamics simulations

        ##region:This computes the forces on the molecules
        def compute_en_force(dftcomm, coords_all, rho: FieldGType, wfn:list[KSWfn] | None = None):
            nonlocal libxc_func, gamma_only, ecut_wfn, e_temp, conv_thr, maxiter, diago_thr_init, iter_printer, mix_beta, mix_dim, ret_vxc
            l_atoms_itr=[]
            num_counter=0
            with dftcomm.image_comm as comm:
                for ityp in range(num_typ):
                    label_sp=label_cryst[ityp]
                    mass_sp=mass_cryst[ityp]
                    ppdata_sp=ppdat_cryst[ityp]
                    num_in_types_sp=num_in_types[ityp]
                    coord_alat_sp=coords_all[num_counter:num_counter+num_in_types_sp]
                    num_counter+=num_in_types_sp
                    Basis_atoms_sp=BasisAtoms.from_cart(label_sp,
                                                    ppdata_sp,
                                                    mass_sp,
                                                    reallat,
                                                    *coord_alat_sp)
                    l_atoms_itr.append(Basis_atoms_sp)
                crystal_itr=Crystal(reallat, l_atoms_itr)
                kpts.recilat= crystal_itr.recilat  ##This is just to circumvent the error message
                
                FieldG_rho_itr: FieldGType= get_FieldG(grho)
                if rho is not None: rho_itr=FieldG_rho_itr(rho.data)
                else: rho_itr=rho

                out = scf(
                        dftcomm=dftcomm, 
                        crystal=crystal_itr, 
                        kpts=kpts, 
                        grho=grho, 
                        gwfn=gwfn,
                        numbnd=numbnd, 
                        is_spin=is_spin, 
                        is_noncolin=is_noncolin, 
                        symm_rho=symm_rho, 
                        rho_start=rho_itr, 
                        wfn_init=wfn, 
                        libxc_func=libxc_func, 
                        occ_typ=occ_typ, 
                        smear_typ=smear_typ, 
                        e_temp=e_temp,  
                        conv_thr=conv_thr, 
                        maxiter=maxiter, 
                        diago_thr_init=diago_thr_init, 
                        iter_printer=iter_printer, 
                        mix_beta=mix_beta, 
                        mix_dim=mix_dim, 
                        dftconfig=dftconfig, 
                        ret_vxc=ret_vxc,
                        force_stress=True
                        )
                
                scf_converged, rho, l_wfn_kgrp, en, v_loc, rho_core, nloc, xc_compute = out
                #region of calculation of the jacobian i.e the force
                force_itr= force(dftcomm=dftcomm,
                                    numbnd=numbnd,
                                    wavefun=l_wfn_kgrp, 
                                    crystal=crystal_itr,
                                    gspc=gwfn,
                                    rho=rho,
                                    vloc=v_loc,
                                    nloc_dij_vkb=nloc,
                                    gamma_only=False,
                                    remove_torque=False,
                                    verbosity=True)[0]

                del v_loc, nloc, xc_compute, crystal_itr, FieldG_rho_itr
            return en.total, force_itr, rho, l_wfn_kgrp
        time=0
        
        rho_md=rho_start
        wfn_md=wfn_init
        ##This is the main loop, all the quantities are in atomic units
        while time<max_t:
            ## Starting of the Iterartion
            en, force_coord, rho_itr, wfn_itr=compute_en_force(dftcomm, coords_cart_all, rho_md, wfn_md)
            en/=RYDBERG
            time_step=int(time/dt)
            if dftcomm.image_comm.rank==0: print("this is iteration", time/dt, "and the force is", force_coord)
            rho_md=rho_itr
            wfn_md=WfnInit(wfn_itr)

            ##SCF calculation is done to calculate the forces, the ground state rho and wavefunctions are used in the next iteration for better convergence.

            accelaration=force_coord.T/mass_all  ##accelaration

            coords_new=2*coords_cart_all-coords_cart_prev+accelaration.T*dt**2  ##Propagation
            d_coord=coords_new-coords_cart_all                                  ##Needed for MSD calculation
            d_COM=np.sum(mass_all*d_coord.T, axis=1)/np.sum(mass_all)           ##Movement of position of COM made zero
            coords_new-=d_COM.T

            ##Calculate the RDist
            RDF_array.append(RDist(crystal, coords_new, num_bins=bins, rmax=rmax)[1])

            ##New velocity and calculation of the velcoity autocorrelation fucntion
            vel_new=(coords_new-coords_cart_prev)/(2*dt)
            vacf=np.sum(vel_new*vel_ref, axis=1)
            vacf=np.mean(vacf, axis=0)/norm_factor

            ##New Kinetic Energy
            ke_new=0.5*np.sum(mass_all*(vel_new.T)**2)
            ##New Temperature
            T_new=2*ke_new/(3*tot_num*BOLTZMANN_RYD)

            del accelaration, force_coord

            ##Total Energy
            en_total=en+ke_new
        
            time_step=int(time/dt)
            time_array[time_step]=time
            temperature_array[time_step]=T_new
            vacf_array[time_step]=vacf

            Ryd_to_eV=27.211386245988/2

            energy_array[time_step]=en_total
            d_coord_ref=coords_new-coords_ref*reallat.alat
            msd_array[time_step]=np.sum(d_coord_ref**2)/tot_num

            del d_coord, d_COM
            
            if dftcomm.image_comm.rank==0:
                print("The total energy of the system is", en_total*Ryd_to_eV, "eV")
                print("The kinetic energy of the system is", ke_new*Ryd_to_eV, "eV")
                print("The potential energy of the system is", en*Ryd_to_eV, "eV")
                print("The temperature of the system is", T_new, "K")
                print("The new coordinates are", coords_new/reallat.alat)
                print("The new MSD is", msd_array[time_step])
                print("The new VACF is", vacf_array[time_step])

            del ke_new, T_new, en_total

            ##printing the energy and the temperature
            
            del coords_cart_prev
            coords_cart_prev=coords_cart_all
            del coords_cart_all
            coords_cart_all=coords_new
            del coords_new

            gc.collect()
            time+=dt
        

        ##End of the iteration
        comm.Bcast(time_array)
        comm.Bcast(temperature_array)
        comm.Bcast(energy_array)
        comm.Bcast(coords_cart_all)
        comm.Bcast(msd_array)


        ##RDF Data
        RDF_array=np.array(RDF_array)
        RDF_mean=np.mean(RDF_array, axis=0)
        RDF_std=np.std(RDF_array, axis=0)
        RDF_data=(r_in, RDF_mean, RDF_std, RDF_array)
        
    return coords_cart_all, time_array, temperature_array , energy_array, msd_array, vacf_array, vel_new, RDF_data
## We have to make a dictionary of the data we want to return.