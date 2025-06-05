from __future__ import annotations
import gc
import tracemalloc
import numpy as numpy

from qtm.constants import RYDBERG, ELECTRONVOLT, vel_HART, BOLTZMANN_SI, BOLTZMANN_HART, M_NUC_HART, MASS_SI
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
qtmconfig.fft_backend = 'pyfftw'


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

tracemalloc.start()

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

    class WfnInit(Protocol):
        def __call__(self, ik: int, kswfn: list[KSWfn]) -> None:
            ...
else:
    IterPrinter = 'IterPrinter'
    WfnInit = 'WfnInit'

## Max_t is the maximum time that can be elapsed, dt is the time steps, and the T_init is the initial temperature of the system.
##If store_var is set to true then, the variables like energy and temperature are stored and 
# these can be plotted with respect to time if is_plot is set to true

def NVE_BEEMAN(dftcomm: DFTCommMod,
          crystal: Crystal,
          max_t: float,
          dt: float,
          T_init: float,
          kpts: KList,
          grho: GSpace,
          gwfn: GSpace,
          ecut_wfn:float,
          numbnd:int,
          is_spin:bool,
          is_noncolin:bool,
          use_symm:bool=False,
          is_time_reversal:bool=False,
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
        mass_cryst=np.array([sp.mass for sp in l_atoms])*M_NUC_HART
        mass_all=np.repeat([sp.mass for sp in l_atoms], [sp.numatoms for sp in l_atoms])*M_NUC_HART
        #tot_mass=np.sum(mass_all)
        num_typ = len(l_atoms)
        reallat=crystal.reallat
        #lnum_labels = np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
        #coords_alat_all = np.concatenate([sp.r_alat for sp in l_atoms], axis=1)
        coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis =1).T
        ##This is a numatom times 3 array containing the coordinates of all the atoms in the crystal
        #endregion: Extract the crystal properties

        ##INIT-MD
        #region: Initializing the Molecular Dynamics simulations
        mass_si=mass_all*MASS_SI
        ##First we assign velocities to the atoms
        #print("the rank of the processor is", comm.rank, "out of the total processors", comm.size)
        vel=np.zeros((tot_num, 3))
        np.random.seed(None)
        vel=np.random.rand(tot_num, 3)-0.5
        #print("random velocities at the time of initialicomzation for processor", comm.rank, vel)
        vel=comm.allreduce(vel)   
        #print("random velocities at the time of initialization for processor after reducing", comm.rank, vel)   
        vel/=comm.size
        ##Calculate the momentum
        #print("velocities at the time of initialization for processor after dividing", comm.rank, vel)
        #print(flush=True)
        momentum=mass_si*vel.T

        ##Subtract the momentum of the center of mass from the momentum of the atoms
        momentum_T=momentum.T
        momentum_T-=np.mean(momentum, axis=1)
        ##Calculate the new velocity after subtracting the momentum of the center of mass
        
        vel=momentum_T.T/mass_si
        print("shapw of vel is", vel.shape)

        ##Calculate the kinetic energy
        ke=0.5*np.sum(mass_si*(vel)**2)
        del momentum, momentum_T
        ##Calculate the temperature
        T=2*ke/(3*tot_num*BOLTZMANN_SI)
        if dftcomm.image_comm.rank==0: print("the temperature calculated from the random velocities is", T, "K")

        ##Rescale the velocities to the desired temperature
        vel*=np.sqrt(T_init/T)

        ##Convert the velocities to atomic units
        vel/=vel_HART
        vel=vel.T
        if dftcomm.image_comm.rank==0: print("re-scaled velocities in atomic units at the time of initializatiion for processor", dftcomm.image_comm.rank, "\n", vel)

        ##Calculate the previous coordinates
        coords_cart_prev=coords_cart_all-vel*dt
        #print("coords cart prev", coords_cart_prev)

        time_step=int(max_t/dt)
        time_array = np.arange(time_step)   
        energy_array = np.empty(shape=time_step)
        temperature_array = np.empty(shape=time_step)

        #endregion: Initializing the Molecular Dynamics simulations

        ##region:This computes the forces on the molecules
        def compute_en_force(dftcomm, coords_all, rho: FieldGType):
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
                        wfn_init=wfn_init, 
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
                
                scf_converged, rho, l_wfn_kgrp, en, v_loc, nloc, xc_compute = out
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
                                    gamma_only=False,
                                    verbosity=True)[0]

                del l_wfn_kgrp, v_loc, nloc, xc_compute, crystal_itr, FieldG_rho_itr
                for var in list(locals().keys()):
                    if var not in ["en", "force_itr", "rho"]:
                        del locals()[var]
                gc.collect()  
                #if dftcomm.image_comm.rank==0:
                    #print("I am process", comm.rank, "and I have calculated the force", force_itr)
            return en.total, force_itr, rho
        time=0

        ##Energy, acceleration in the previous step
        rho_md=rho_start
        en_prev, force_prev, rho_prev=compute_en_force(dftcomm, coords_cart_prev, rho_md)
        force_prev*=RYDBERG
        accl_prev=force_prev.T/mass_all
        rho_md=rho_prev
        del force_prev, rho_prev

        ##Energy, acceleration in the current time step
        en_present, force_present, rho_present=compute_en_force(dftcomm, coords_cart_all, rho_md)
          ##Energy and force are calculated in Rydberg units, convert these to Hartree units
        force_present*=RYDBERG
        accl_present=force_present.T/mass_all  ## Calculating accelaration
        rho_md=rho_present
        vel_present=vel
        ke_present=0.5*np.sum(mass_all*(vel_present.T)**2)
        T_present=2*ke_present/(3*tot_num*BOLTZMANN_HART)
        if dftcomm.image_comm.rank==0:
            print("the temperature is", T_present, "K")
        en_total=en_present+ke_present
        time_step=int(time)
        temperature_array[time_step]=T_present
        energy_array[time_step]=en_total
        time+=dt
        del en_present, force_present, rho_present, ke_present, T_present, en_total
        ##This is the main loop, all the quantities are in atomic units
        while time<max_t:
            ##Predictor step
            print("shape of vel_present is", vel_present.shape)
            coords_new=coords_cart_all + vel_present*dt + (1/6)*(4*accl_present.T-accl_prev.T)*dt**2  ##predict the new coordinates 

            d_coord=coords_new-coords_cart_all
            d_COM=np.sum(mass_all*d_coord.T, axis=1)/np.sum(mass_all)
            coords_new-=d_COM.T
            del d_coord, d_COM

            en_new, force_new, rho_new=compute_en_force(dftcomm, coords_cart_all, rho_md)
            if dftcomm.image_comm.rank==0: print("this is iteration", time/dt, "and the force is", force_new)
            del rho_md
            rho_md=rho_new
            del rho_new
            ##Energy and force are calculated in Rydberg units, convert these to Hartree units
            force_new*=RYDBERG
            acc_new=force_new.T/mass_all

            ##Corrector step
            coords_new=coords_cart_all + vel_present*dt + (1/6)*(acc_new.T+2*accl_present.T)*dt**2
            d_coord=coords_new-coords_cart_all
            d_COM=np.sum(mass_all*d_coord.T, axis=1)/np.sum(mass_all)
            coords_new-=d_COM.T
            vel_new=(coords_new-coords_cart_prev)/dt + (1/6)*(2*acc_new.T+accl_present.T)*dt
            del d_coord, d_COM

            if dftcomm.image_comm.rank==0:
                print("change in coordinates in each iteration", coords_new-coords_cart_all)

            ##New velocity
            ##Step-3, update of position, velocity and acceleration
            coords_cart_all=coords_new
            vel_present=vel_new
            accl_prev=accl_present
            del accl_present, coords_new
            accl_present=acc_new
            
            ##New Kinetic Energy
            #ke_new=0.5*np.sum(np.sum(momentum_new**2, axis=1)/mass_all)
            ke_new=0.5*np.sum(mass_all*(vel_new.T)**2)
            ##New Temperature
            T_new=2*ke_new/(3*tot_num*BOLTZMANN_HART)

            del force_new, vel_new

            ##Total Energy
            en_total=en_new+ke_new
        
            time_step=int(time/dt)
            time_array[time_step]=time
            temperature_array[time_step]=T_new


            Har_to_eV=27.211386245988
            energy_array[time_step]=en_total
            

            if dftcomm.image_comm.rank==0:
                print("The total energy of the system is", en_total*Har_to_eV, "eV")
                print("The kinetic energy of the system is", ke_new*Har_to_eV, "eV")
                print("The potential energy of the system is", en_new*Har_to_eV, "eV")
                print("The temperature of the system is", T_new, "K")
                print("The new coordinates are", coords_cart_all/reallat.alat)

            del en_new, ke_new, T_new, en_total

            ##printing the energy and the temperature
            
            print(flush=True)
            gc.collect()
            time+=dt

            
            ##End of the iteration

        time_array=np.array(time_array)
        temperature_array=np.array(temperature_array)
        energy_array=np.array(energy_array)
        
        comm.Bcast(time_array)
        comm.Bcast(temperature_array)
        comm.Bcast(energy_array)
        comm.Bcast(coords_cart_all)
        
    return coords_cart_all, time_array, temperature_array , energy_array

    
