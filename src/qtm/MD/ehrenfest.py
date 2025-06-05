from typing import Optional, Callable

from __future__ import annotations
from typing import TYPE_CHECKING
import logging

from qtm.logger import qtmlogger
from qtm.config import MPI4PY_INSTALLED

import numpy as np
from qtm.config import qtmconfig
from qtm.containers.field import FieldGType, FieldRType
from qtm.containers.wavefun import WavefunGType
from qtm.crystal.crystal import Crystal, BasisAtoms
from qtm.dft.kswfn import KSWfn
from qtm.dft.scf import EnergyData
from qtm.gspace import GSpace, GkSpace
from qtm.mpi.comm import QTMComm
from qtm.pot import ewald, hartree, xc
from qtm.pot.xc import check_libxc_func, get_libxc_func
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.pseudo.nloc import NonlocGenerator
from qtm.tddft_gamma.prop.etrs import normalize_rho
from tqdm import trange
from qtm.force import force
from qtm.dft.config import DFTCommMod
from qtm.constants import RYDBERG, ELECTRONVOLT, vel_HART, BOLTZMANN_SI, BOLTZMANN_HART, M_NUC_HART, MASS_SI

if MPI4PY_INSTALLED:
    from qtm.mpi.containers import get_DistFieldG
from qtm.mpi.gspace import DistGSpace, DistGkSpace

if TYPE_CHECKING:
    from typing import Literal
    from numbers import Number
__all__ = ["scf", "EnergyData", "Iterprinter"]

def ehrenfest(
    dftcomm: DFTCommMod,
    crystal: Crystal,
    rho_start: FieldGType,
    wfn_gamma: list[list[KSWfn]],
    T_init: float,
    occ_typ: Literal["smear", "fixed"],
    time_step: float,
    numstep: int,
    dipole_updater: Callable[[int, FieldGType, WavefunGType], None],
    libxc_func: Optional[tuple[str, str]] = None,
    
):
    # Begin setup ========================================
    ##Setting up the crystal parameters
    l_atoms = crystal.l_atoms
    tot_num = np.sum([sp.numatoms for sp in l_atoms])
    num_in_types=np.array([sp.numatoms for sp in l_atoms])
    ppdat_cryst=np.array([sp.ppdata for sp in l_atoms])
    label_cryst=np.array([sp.label for sp in l_atoms])
    mass_cryst=np.array([sp.mass for sp in l_atoms])*M_NUC_HART
    mass_all=np.repeat([sp.mass for sp in l_atoms], [sp.numatoms for sp in l_atoms]).reshape(-1,1)*M_NUC_HART
    tot_mass=np.sum(mass_all)
    num_typ = len(l_atoms)
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis =1).T


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
    dftcomm.image_comm.Bcast(vel)      
    ##Calculate the momentum
    momentum=mass_si*vel

    ##Compute the momentum of the center of mass
    momentum_cm=np.sum(momentum, axis=0)

    ##Subtract the momentum of the center of mass from the momentum of the atoms
    momentum-=momentum_cm

    ##Calculate the new velocity after subtracting the momentum of the center of mass
    
    vel=momentum/mass_si

    ##Calculate the kinetic energy
    ke=0.5*np.sum(np.sum(momentum**2, axis=1)/mass_si)

    ##Calculate the temperature
    T=2*ke/(3*tot_num*BOLTZMANN_SI)
    #print("the temperature calculated from the random velocities is", T, "K")

    ##Rescale the velocities to the desired temperature
    vel*=np.sqrt(T_init/T)

    ##Convert the velocities to atomic units
    vel/=vel_HART
    #print("re-scaled velocities in atomic units are", vel)

    ##Calculate the previous coordinates
    coords_cart_prev=coords_cart_all-vel*time_step
    #print("coords cart prev", coords_cart_prev)

    time_array=[]
    energy_array=[]
    temperature_array=[]

    ## Starting of the loops
    ##Making the Hamiltonian Propagator  from the crystal:
    def prop_ehrenfest(crystal: Crystal, rho: FieldGType):
        ##Extracting the parameters from the crystal
        reallat=crystal.reallat
        coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis =1).T
        v_ion = rho.zeros(1)
        rho_core = rho.zeros(1)
        v_ion_list=[]
        l_nloc = []
        for sp in crystal.l_atoms:
            v_ion_typ, rho_core_typ, v_ion_nomult, rho_core_nomult = loc_generate_pot_rhocore(sp, gspc_rho)
            v_ion += v_ion_typ
            rho_core += rho_core_typ
            l_nloc.append(NonlocGenerator(sp, gspc_wfn))
            v_ion_list.append(v_ion_nomult)
        v_ion = v_ion.to_r()

        v_hart: FieldRType
        v_xc: FieldRType
        v_loc: FieldRType
        
        ##Function to compute the parameters like local potential and the total energy
        def compute_pot_local(rho_):
            nonlocal v_loc, v_xc, v_hart
            nonlocal rho_core
            v_hart, en.hartree = hartree.compute(rho_)
            v_xc, en.xc, GGA = xc.compute(rho_, rho_core, *libxc_func)
            v_loc = v_ion + v_hart + v_xc
            dftcomm.image_comm.bcast(v_loc._data)
            v_loc *= 1 / np.prod(gspc_wfn.grid_shape)
            return v_loc
        
        en: EnergyData = EnergyData()
        def compute_en(rho_):
            if occ_typ == "smear":
                en.fermi, en.smear, en.internal = 0, 0, 0
            elif occ_typ == "fixed":
                en.HO_level, en.LU_level = 0, 0
            e_eigen=0
            for kswfn_k in wfn_gamma[0]:
                e_eigen += sum(
                    kswfn_.k_weight * np.sum(kswfn_.evl * kswfn_.occ)
                    for kswfn_ in kswfn_k
                )
            v_hxc=v_hart+v_xc
            e_vhxc = sum((v_hxc * rho_.to_r())).integrate_unitcell()
            en.one_el = (e_eigen - e_vhxc).real
            en.total = en.one_el + en.ewald + en.hartree + en.xc
            if occ_typ == "smear":
                en.total += en.smear
                en.hwf += en.smear
                en.internal = en.total - en.smear
        if libxc_func is None:
            libxc_func = xc.get_libxc_func(crystal)
        else:
            xc.check_libxc_func(libxc_func)
    # End setup ========================================

    # Select expoper and propagator =====================
    #region selecting the methods expoper and propagator
        prop_kwargs = {}
        print(f"config.tddft_exp_method: {config.tddft_exp_method}")
        if config.tddft_exp_method == "taylor":
            from qtm.tddft_gamma.expoper.taylor import TaylorExp as PropOper
            prop_kwargs["order"] = config.taylor_order
        elif config.tddft_exp_method == "splitoper":
            from qtm.tddft_gamma.expoper.splitoper import SplitOper as PropOper
        else:
            raise ValueError(
                "'config.tddft_exp_method' not recognized. "
                f"got {config.tddft_exp_method}."
            )

        if config.tddft_prop_method == "etrs":
            from qtm.tddft_gamma.prop.etrs import prop_step
        elif config.tddft_prop_method == "splitoper":
            if config.tddft_exp_method != "splitoper":
                raise ValueError(
                    "'config.tddft_exp_method' must be 'splitoper' when "
                    "'config.tddft_prop_method' is set to 'splitoper. "
                    f"got {config.tddft_exp_method} instead."
                )
            from qtm.tddft_gamma.prop.splitoper import prop_step
        else:
            raise ValueError(
                "'config.tddft_prop_method' not recognized. "
                f"got {config.tddft_prop_method}."
            )
    #endregion
        # End selecting expoper and propagator ==============

        # Begin propagation ================================
        v_loc = compute_pot_local(rho, rho_core, v_ion)
        prop_gamma = PropOper(
            gkspc, is_spin, is_noncolin, v_loc, l_nloc, time_step, **prop_kwargs
        )
        force_itr=force(dftcomm=dftcomm,
                        numbnd=numbnd,
                        wavefun=wfn_gamma, 
                        crystal=crystal,
                        gspc=gspc_wfn,
                        rho=rho,
                        vloc=v_loc,
                        nloc_dij_vkb=l_nloc,
                        gamma_only=False,
                        verbosity=False)[0]
        
        prop_step(wfn_gamma, rho, crystal.numel, compute_pot_local, prop_gamma)
        rho = (
            wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
        )
        normalize_rho(rho, numel)
        compute_en(rho)

        #region: Propagation of the coordinates
        #Finding the new coordinates
        accelaration=force_itr/mass_all
        coords_new=2*coords_cart_all-coords_cart_prev+accelaration*time_step**2
        vel_new=(coords_new-coords_cart_prev)/(2*time_step)

        ##New momentum
        momentum_new=mass_all*vel_new

        ##New Kinetic Energy
        ke_new=0.5*np.sum(np.sum(momentum_new**2, axis=1)/mass_all)

        ##New Temperature
        T_new=2*ke_new/(3*tot_num*BOLTZMANN_HART)

        ##Total Energy
        en_total=en.total+ke_new
        energy_array.append(en_total)
        temperature_array.append(T_new)
        #endregion

        #region: Make a crystal with the new coordinates
        l_atoms_itr=[]
        num_counter=0
        with dftcomm.image_comm as comm:
            for ityp in range(num_typ):
                label_sp=label_cryst[ityp]
                mass_sp=mass_cryst[ityp]
                ppdata_sp=ppdat_cryst[ityp]
                num_in_types_sp=num_in_types[ityp]
                coord_alat_sp=coords_cart_all[num_counter:num_counter+num_in_types_sp]
                num_counter+=num_in_types_sp
                Basis_atoms_sp=BasisAtoms.from_cart(label_sp,
                                                ppdata_sp,
                                                mass_sp,
                                                reallat,
                                                *coord_alat_sp)
                l_atoms_itr.append(Basis_atoms_sp)
            crystal=Crystal(reallat, l_atoms_itr)
        #endregion

    rho = rho_start.copy()
    for istep in trange(numstep):
        # FIXME: We are implicitly assuming that the wavefunction is not spin-polarized.
        prop_ehrenfest(crystal, rho)
        time_array.append(istep*time_step)

    # End propagation ================================
