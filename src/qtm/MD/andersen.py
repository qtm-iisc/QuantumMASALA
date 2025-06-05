from __future__ import annotations
import gc

import numpy as numpy

from qtm.constants import RYDBERG, ELECTRONVOLT, vel_HART, BOLTZMANN_SI, BOLTZMANN_RYD, M_NUC_RYD
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger
qtmconfig.fft_backend = 'mkl_fft'

from typing import TYPE_CHECKING

from qtm.logger import qtmlogger
if TYPE_CHECKING:
    from typing import Literal
    from numbers import Number
__all__ = ['scf', 'EnergyData', 'IterPrinter']

from dataclasses import dataclass
from time import perf_counter
from sys import version_info
import numpy as np

from qtm.crystal import Crystal
from qtm.kpts import KList
from qtm.gspace import GSpace, GkSpace
from qtm.mpi.gspace import DistGSpace
from qtm.containers import FieldGType, FieldRType, get_FieldG

from qtm.pot import hartree, xc, ewald
from qtm.pseudo import (
    loc_generate_rhoatomic, loc_generate_pot_rhocore,
    NonlocGenerator
)
from qtm.symm.symmetrize_field import SymmFieldMod

from qtm.dft import DFTCommMod, DFTConfig, KSWfn, KSHam, eigsolve, occup, mixing

from qtm.mpi.check_args import check_system
from qtm.mpi.utils import scatter_slice

from qtm.force import force

from qtm.msg_format import *
from qtm.constants import RYDBERG

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)

@dataclass
class EnergyData:
    total: float = 0.0
    hwf: float = 0.0
    one_el: float = 0.0
    ewald: float = 0.0
    hartree: float = 0.0
    xc: float = 0.0

    fermi: float | None = None
    smear: float | None = None
    internal: float | None = None

    HO_level: float | None = None
    LU_level: float | None = None


if version_info[1] >= 8:
    from typing import Protocol

    class IterPrinter(Protocol):
        def __call__(self, idxiter: int, runtime: float, scf_converged: bool,
                     e_error: float, diago_thr: float, diago_avgiter: float,
                     en: EnergyData) -> None:
            ...

    class WfnInit:
        def __init__(self, pre_existing_wfns: list[KSWfn]):
            self.pre_existing_wfns = pre_existing_wfns

        def __call__(self, ik: int, kswfn: list[KSWfn]) -> None:
            """Initialize wavefunctions using pre-existing wavefunctions.
            For now it only works for spin unpolarised case"""
            assert len(kswfn) == len(self.pre_existing_wfns[ik])
            for i in range(len(kswfn)):
                kswfn[i].evc_gk.data[:] = self.pre_existing_wfns[ik][i].evc_gk.data
                kswfn[i].evl[:]= self.pre_existing_wfns[ik][i].evl[:]
                kswfn[i].occ[:]= self.pre_existing_wfns[ik][i].occ[:]  
else:
    IterPrinter = 'IterPrinter'
    WfnInit = 'WfnInit'

## Max_t is the maximum time that can be elapsed, dt is the time steps, and the T_init is the initial temperature of the system.
##If store_var is set to true then, the variables like energy and temperature are stored and 
# these can be plotted with respect to time if is_plot is set to true
def Andersen_MD(dftcomm: DFTCommMod,
          crystal: Crystal,
          max_t: float,
          dt: float,
          T_init: float,
          vel_init: None | np.ndarray,
          nu: float,
          kpts:KList,
          grho: GSpace,
          gwfn: GSpace,
          ecut_wfn:float,
          numbnd:int,
          is_spin:bool,
          is_noncolin:bool,
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
        l_atoms = crystal.l_atoms
        tot_num = np.sum([sp.numatoms for sp in l_atoms])
        num_in_types=np.array([sp.numatoms for sp in l_atoms])
        ppdat_cryst=np.array([sp.ppdata for sp in l_atoms])
        label_cryst=np.array([sp.label for sp in l_atoms])
        mass_cryst=np.array([sp.mass for sp in l_atoms])*M_NUC_RYD
        mass_all=np.repeat([sp.mass for sp in l_atoms], [sp.numatoms for sp in l_atoms])*M_NUC_RYD
        #tot_mass=np.sum(mass_all)
        num_typ = len(l_atoms)
        reallat=crystal.reallat
        #lnum_labels = np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
        #coords_alat_all = np.concatenate([sp.r_alat for sp in l_atoms], axis=1)
        coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis =1).T
        coods_ref=coords_cart_all
        ##This is a numatom times 3 array containing the coordinates of all the atoms in the crystal


        ##INIT-MD
        ##First we assign velocities to the atoms
        if type(vel_init) is None:
            vel=np.random.rand(tot_num, 3)-0.5
            vel=comm.allreduce(vel)
            vel/=comm.size
            ##Calculate the momentum
            momentum=mass_all*vel.T
            momentum_T=np.array(momentum).T

            ##Compute the momentum of the center of mass
            momentum_T-=np.mean(momentum_T, axis=0)


            ##Calculate the updated velocity of the atoms
            vel=momentum_T.T/mass_all
            vel=vel.T

            ##Calculate the kinetic energy
            ke_init=0.5*np.sum(mass_all*vel.T**2)
            del momentum, momentum_T
            ##Calculate the temperature
            T=2*ke_init/(3*tot_num*BOLTZMANN_RYD)

            print("Initially the temperature from the random velocities are", T, "K")  

            ##Rescale the velocities
            # to the desired temperature
            vel*=np.sqrt(T_init/T)
        else:
            vel= vel_init

        #region Debug statement
        ##Calculate the kinetic energy
        ke_init=0.5*np.sum(mass_all*vel.T**2)

        ##Calculate the temperature
        T_later=2*ke_init/(3*tot_num*BOLTZMANN_RYD)

        print("After rescaling the temperature is", T_later, "K")
        #endregion End of debug statement

        ##Convert the velocities to atomic units

        time_step=int(max_t/dt)
        time_array = np.empty(time_step)
        energy_array = np.empty(time_step)
        ke_array=np.empty(time_step)
        pe_array=np.empty(time_step)
        temperature_array = np.empty(time_step)
        msd_array = np.empty(time_step)

        ##This computes the forces on the molecules
        def compute_en_force(dftcomm, coords_all, rho: FieldGType, wfn:list[KSWfn]| None=None):
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
                #kpts_itr=gen_monkhorst_pack_grid(crystal_itr, kgrid, kshift, use_symm, is_time_reversal)
                
                FieldG_rho_itr: FieldGType= get_FieldG(grho)
                if rho is not None: rho_itr=FieldG_rho_itr(rho.data)
                else: rho_itr=rho

                '''with dftcomm.image_comm as comm: 
                    print("Hello! my rank is, ", comm.rank)
                    print("the primvector I have in my lattice is", crystal_itr.reallat.primvec)
                    print(flush=True)'''

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
                
                scf_converged, rho, l_wfn_kgrp, en, v_loc, nloc, xc_compute, del_vhxc = out
                if comm.rank==0:  
                    print("my rank is", dftcomm.image_comm.rank)
                    print("And I have successfully calculated energy", en.total)
                #region of calculation of the jacobian i.e the force

                force_itr= force(dftcomm=dftcomm,
                                    numbnd=numbnd,
                                    wavefun=l_wfn_kgrp, 
                                    crystal=crystal_itr,
                                    gspc=gwfn,
                                    rho=rho,
                                    vloc=v_loc,
                                    nloc_dij_vkb=nloc,
                                    del_v_hxc=del_vhxc,
                                    gamma_only=False,
                                    verbosity=True)[0]

                del v_loc, nloc, xc_compute, crystal_itr, FieldG_rho_itr
                for var in list(locals().keys()):
                    if var not in ["en", "force_itr", "rho"]:
                        del locals()[var]
                gc.collect()  
                #if dftcomm.image_comm.rank==0:
                    #print("I am process", comm.rank, "and I have calculated the force", force_itr)
                energy=en.total/RYDBERG
            return energy, force_itr, rho, l_wfn_kgrp
        
        #Initial configuration of the system
        en, force_coord, rho, wfn_in=compute_en_force(dftcomm, coords_cart_all, rho_start, wfn=wfn_init)
        force_coord
        rho_md=rho
        wfn_md=WfnInit(wfn_in)
        time=0
        while time<max_t:
            ## Starting of the Iterartion
            ## Calculating accelaration
            accelaration=force_coord.T/mass_all
            #updating position
            coords_new=coords_cart_all + vel*dt+ 0.5*accelaration.T*dt**2
            d_coord=coords_new-coords_cart_all
            d_COM=np.sum(mass_all*d_coord.T, axis=1)/np.sum(mass_all)
            coords_new-=d_COM.T

            del d_coord, d_COM

            d_coord_ref=coords_new-coods_ref
            msd=np.sum(d_coord_ref**2)
            ##Calculating the new energy and forces
            en_new, force_coord_new, rho, wfn=compute_en_force(dftcomm, coords_new, rho_md, wfn_md)
            rho_md=rho
            wfn_md=WfnInit(wfn)
            ##Calculating the new velocity
            #region debug statement
            
            #endregion end of debug statement
            vel=vel+0.5*(accelaration+force_coord_new.T/mass_all).T*dt

            #reassigning the forces and coordinates
            force_coord=force_coord_new
            coords_cart_all=coords_new

            ###Application of the Andersen Thermostat
            kT=T_init*BOLTZMANN_RYD

            if comm.rank==0: print("velocity before random scaling", vel)

            #region debug statement for the velocity
            #Calculating the total energy and Temperature
            ke=0.5*np.sum(mass_all*vel.T**2)
            en_total=en_new+ke
            T_new=2*ke/(3*tot_num*BOLTZMANN_RYD)

            del accelaration, coords_new, force_coord_new



            ##printing all the variables
            #region debug statement
            if comm.rank==0:

                print("the time is", time)
                print("the temperature before scaling is", T_new)
                print("the energy before scaling is ", en_total)
                print("the ke energy before scaling is", ke)
            #endregion end of debug statement

            prob=float(nu)*dt
            print("the probability is", prob)

            if comm.rank==0:
                for atom in range(tot_num):
                    b=np.random.rand()
                    print("the random number is", b)
                    if b <prob:
                        sigma=np.sqrt(kT/mass_all[atom])
                        vel[atom]=np.random.normal(0, sigma, 3) ##This is in SI units
                print("velocity after random scaling", vel)
            comm.Bcast(vel)
            print("the velocity after the random scaling is", vel)

            ##Make the center of Mass stationary
            momentum=mass_all*vel.T
            momentum_T=np.array(momentum).T
            ##Compute the momentum of the center of mass
            momentum_T-=np.mean(momentum_T, axis=0)
            ##Calculate the updated velocity of the atoms
            vel=momentum_T.T/mass_all
            vel=vel.T
            ##Calculate the kinetic energy


            #Calculating the total energy and Temperature
            ke=0.5*np.sum(mass_all*vel.T**2)
            en_total=en_new+ke
            T_new=2*ke/(3*tot_num*BOLTZMANN_RYD)

            ##printing all the variables
            #region debug statement
            if comm.rank==0:
                print("the time is", time)
                print("the temperature is", T_new)
                print("the energy is ", en_total)
                print("the ke energy is", ke)
            #endregion end of debug statement

            time_step=int(time/dt)
            Ryd_to_eV=27.211386245988/2
            time_array[time_step]=time
            energy_array[time_step]=en_total*Ryd_to_eV
            ke_array[time_step]=ke*Ryd_to_eV
            pe_array[time_step]=en_new*Ryd_to_eV
            temperature_array[time_step]=T_new
            msd_array[time_step]=msd

            if dftcomm.image_comm.rank==0:
                print("The total energy of the system is", en_total*Ryd_to_eV, "eV")
                print("The kinetic energy of the system is", ke*Ryd_to_eV, "eV")
                print("The potential energy of the system is", en*Ryd_to_eV, "eV")
                print("The temperature of the system is", T_new, "K")
                print("The new coordinates are", coords_cart_all/reallat.alat)
                print("The new MSD is", msd_array[time_step])

            time+=dt

        time_array=np.array(time_array)
        temperature_array=np.array(temperature_array)
        energy_array=np.array(energy_array)
        ke_array=np.array(ke_array)
        pe_array=np.array(pe_array)
        msd_array=np.array(msd_array)

        comm.Bcast(time_array)
        comm.Bcast(temperature_array)
        comm.Bcast(energy_array)
        comm.Bcast(coords_cart_all)

    return coords_cart_all, time_array, temperature_array , energy_array, ke_array, pe_array, msd_array, vel

    