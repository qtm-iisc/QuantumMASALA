from __future__ import annotations
import gc
import tracemalloc
import numpy as numpy

from qtm.constants import RYDBERG, ELECTRONVOLT, vel_RYD, BOLTZMANN_SI, BOLTZMANN_RYD, M_NUC_RYD, MASS_SI
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
        #print(mass_all)
        #tot_mass=np.sum(mass_all)
        num_typ = len(l_atoms)
        reallat=crystal.reallat
        #lnum_labels = np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
        #coords_alat_all = np.concatenate([sp.r_alat for sp in l_atoms], axis=1)
        coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis =1).T
        coords_ref=coords_cart_all/reallat.alat
        ##This is a numatom times 3 array containing the coordinates of all the atoms in the crystal
        #endregion: Extract the crystal properties

        ##INIT-MD
        #region: Initializing the Molecular Dynamics simulations
        #mass_si=mass_all*MASS_SI
        ##First we assign velocities to the atoms
        #print("the rank of the processor is", comm.rank, "out of the total processors", comm.size)
        vel=np.zeros((tot_num, 3))
        np.random.seed(None)
        if type(vel_init)==None:
            vel=np.random.rand(tot_num, 3)-0.5
            #print("random velocities at the time of initialicomzation for processor", comm.rank, vel)
            vel=comm.allreduce(vel)   
            #print("random velocities at the time of initialization for processor after reducing", comm.rank, vel)   
            vel/=comm.size
            #print("the velocities at the time of initialization for processor", comm.rank, vel)
            #print("the velocities at the time of initialization for processor after reducing", comm.rank, vel)
            ##Calculate the momentum
            #print("velocities at the time of initialization for processor after dividing", comm.rank, vel)
            #print(flush=True)
            momentum=mass_all*vel.T

            ##Subtract the momentum of the center of mass from the momentum of the atoms
            momentum_T=momentum.T
            momentum_T-=np.mean(momentum, axis=1)
            ##Calculate the new velocity after subtracting the momentum of the center of mass
            
            vel=momentum_T.T/mass_all
            #print("vel", vel)
            #print("shapw of vel is", vel.shape)

            ##Calculate the kinetic energy
            ke=0.5*np.sum(mass_all*(vel)**2)
            del momentum, momentum_T
            ##Calculate the temperature
            T=2*ke/(3*tot_num*BOLTZMANN_RYD)
        # print("BOLTZMANN_RYD", BOLTZMANN_RYD)
            #if dftcomm.image_comm.rank==0: print("the temperature calculated from the random velocities is", T, "K")
            ##Rescale the velocities to the desired temperature
            vel*=np.sqrt(T_init/T)
        else:
            vel=vel_init.T
        vel_ref=vel.T
        norm_factor=np.mean(np.sum(vel**2, axis=1))
        ke_scale=0.5*np.sum(mass_all*(vel**2))
        T_scale=2*ke_scale/(3*tot_num*BOLTZMANN_RYD)
        if dftcomm.image_comm.rank==0: print("the temperature after rescaling is", T_scale, "K")
        ##Convert the velocities to atomic units
        if dftcomm.image_comm.rank==0: print("re-scaled velocities in atomic units at the time of initializatiion for processor", dftcomm.image_comm.rank, "\n", vel)
       ##Calculate the previous coordinates
        coords_cart_prev=coords_cart_all-vel.T*dt
        #print("coords cart prev", coords_cart_prev)
        del vel

        time_step=int(max_t/dt)
        time_array = np.empty(time_step)
        energy_array = np.empty(time_step)
        temperature_array = np.empty(time_step)
        msd_array= np.empty(time_step)
        vacf_array= np.empty(time_step)

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
                '''if comm.rank==0:  
                    print("my rank is", dftcomm.image_comm.rank)
                    print("And I have successfully calculated energy", en.total)'''
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
                                    remove_torque=False,
                                    verbosity=True)[0]

                del v_loc, nloc, xc_compute, crystal_itr, FieldG_rho_itr
                for var in list(locals().keys()):
                    if var not in ["en", "force_itr", "rho"]:
                        del locals()[var]
                gc.collect()  
                #if dftcomm.image_comm.rank==0:
                    #print("I am process", comm.rank, "and I have calculated the force", force_itr)
            return en.total, force_itr, rho, l_wfn_kgrp
        time=0
        rho_md=rho_start
        wfn_md=wfn_init
        ##This is the main loop, all the quantities are in atomic units
        while time<max_t:
            ## Starting of the Iterartion
            en, force_coord, rho_itr, wfn_itr=compute_en_force(dftcomm, coords_cart_all, rho_md, wfn_md)
            
            #print("en", en)
            ##The energy in Hartree units
            en/=RYDBERG
            if dftcomm.image_comm.rank==0: print("this is iteration", time/dt, "and the force is", force_coord)
            del rho_md, wfn_md
            rho_md=rho_itr
            wfn_md=WfnInit(wfn_itr)
            del rho_itr
            ##Energy and force are calculated in Rydberg units, convert these to Hartree units
            ## Calculating accelaration
            accelaration=force_coord.T/mass_all

            coords_new=2*coords_cart_all-coords_cart_prev+accelaration.T*dt**2
            d_coord=coords_new-coords_cart_all
            d_COM=np.sum(mass_all*d_coord.T, axis=1)/np.sum(mass_all)
            coords_new-=d_COM.T

            

            if dftcomm.image_comm.rank==0:
                print("change in coordinates in each iteration", coords_new-coords_cart_all)

            ##New velocity
            vel_new=(coords_new-coords_cart_prev)/(2*dt)
            vacf=np.sum(vel_new*vel_ref, axis=1)
            vacf=np.mean(vacf, axis=0)/norm_factor

            if dftcomm.image_comm.rank==0:
                print("the new velocity at this iteration is", vel_new/reallat.alat)
            
            ##New Kinetic Energy
            #ke_new=0.5*np.sum(np.sum(momentum_new**2, axis=1)/mass_all)
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

            del en, ke_new, T_new, en_total

            ##printing the energy and the temperature
            
            del coords_cart_prev
            coords_cart_prev=coords_cart_all
            del coords_cart_all
            coords_cart_all=coords_new
            del coords_new
            
            print(flush=True)
            gc.collect()
            time+=dt
        
        comm.Bcast(time_array)
        comm.Bcast(temperature_array)
        comm.Bcast(energy_array)
        comm.Bcast(coords_cart_all)
        comm.Bcast(msd_array)
        
    return coords_cart_all, time_array, temperature_array , energy_array, msd_array, vacf_array, vel_new