
from __future__ import annotations


import numpy as np
from scipy.optimize import minimize

from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_scf_status

from qtm.config import qtmconfig
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
from qtm.config import NDArray

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

    class WfnInit(Protocol):
        def __call__(self, ik: int, kswfn: list[KSWfn]) -> None:
            ...
else:
    IterPrinter = 'IterPrinter'
    WfnInit = 'WfnInit'


def relax(dftcomm: DFTCommMod,
          constraint:np.ndarray,
          crystal: Crystal,
          kgrid:tuple[int, int, int],
          kshift: tuple[bool, bool, bool],
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
          gamma_only:bool=False,
          ):
    
    l_atoms = crystal.l_atoms
    #tot_num = np.sum([sp.numatoms for sp in l_atoms])
    num_in_types=[sp.numatoms for sp in l_atoms]
    ppdat_cryst=[sp.ppdata for sp in l_atoms]
    label_cryst=[sp.label for sp in l_atoms]
    mass_cryst=[sp.mass for sp in l_atoms]
    num_typ = len(l_atoms)
    reallat=crystal.reallat
    #lnum_labels = np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    #coords_alat_all = np.concatenate([sp.r_alat for sp in l_atoms], axis=1)
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    coords_cart_all_flat=coords_cart_all.T.flatten()


    def compute_en_jac(coords_flatten):
        nonlocal libxc_func, gamma_only
        coords_alat_itr=coords_flatten.reshape(-1,3)
        l_atoms_itr=[]
        num_counter=0
        for ityp in range(num_typ):
            label_sp=label_cryst[ityp]
            mass_sp=mass_cryst[ityp]
            ppdata_sp=ppdat_cryst[ityp]
            num_in_types_sp=num_in_types[ityp]
            coord_alat_sp=coords_alat_itr[num_counter:num_counter+num_in_types_sp]
            num_counter+=num_in_types_sp
            Basis_atoms_sp=BasisAtoms.from_cart(label_sp,
                                               ppdata_sp,
                                               mass_sp,
                                               reallat,
                                               *coord_alat_sp)
            l_atoms_itr.append(Basis_atoms_sp)
        crystal_itr=Crystal(reallat, l_atoms_itr)
        kpts_itr=gen_monkhorst_pack_grid(crystal_itr, kgrid, kshift, use_symm, is_time_reversal)

        ecut_rho=4*ecut_wfn
        grho_itr_serial=GSpace(crystal_itr.recilat, ecut_rho)
        if dftcomm.n_pwgrp == dftcomm.image_comm.size:  
            grho_itr = grho_itr_serial
        else:
            grho_itr = DistGSpace(comm_world, grho_itr_serial)
        gwfn_itr=grho_itr
                #FieldG_rho_itr: FieldGType= get_FieldG(grho_itr)
        out = scf(
                dftcomm=dftcomm, 
                crystal=crystal_itr, 
                kpts=kpts_itr, 
                grho=grho_itr, 
                gwfn=gwfn_itr,
                numbnd=numbnd, 
                is_spin=is_spin, 
                is_noncolin=is_noncolin, 
                symm_rho=symm_rho, 
                rho_start=rho_start, 
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
        
        scf_converged, rho, l_wfn_kgrp, en, v_loc, nloc, xc_compute= out

        print(flush=True)
        '''wfn_relax=l_wfn_kgrp
        rho_relax=rho'''

        force_itr= force(dftcomm=dftcomm,
                         numbnd=numbnd,
                         wavefun=l_wfn_kgrp, 
                         crystal=crystal_itr,
                         gspc=gwfn_itr,
                         rho=rho,
                         vloc=v_loc,
                         nloc_dij_vkb=nloc,
                         gamma_only=False,
                         verbosity=True)[0]
        
        force_itr*=constraint[:, None]
        jac=-force_itr.flatten()
        print("forces are", force_itr)
        print(flush=True)
        return en.total, jac


    minimization= minimize(compute_en_jac, coords_cart_all_flat, method='BFGS', jac=True, options={'disp': True, 'gtol': 1E-7})

    ## The final coordinates
    coords_cart_final=minimization.x

    ##The final energy
    en_final=minimization.fun

   
    coords_cart_final_itr=coords_cart_final.reshape(-1,3)
    l_atoms_itr=[]
    num_counter=0
    for ityp in range(num_typ):
        label_sp=label_cryst[ityp]
        mass_sp=mass_cryst[ityp]
        ppdata_sp=ppdat_cryst[ityp]
        num_in_types_sp=num_in_types[ityp]
        coord_cart_sp=coords_cart_final_itr[num_counter:num_counter+num_in_types_sp]
        num_counter+=num_in_types_sp
        Basis_atoms_sp=BasisAtoms.from_cart(label_sp,
                                            ppdata_sp,
                                            mass_sp,
                                            reallat,
                                            *coord_cart_sp)
        l_atoms_itr.append(Basis_atoms_sp)
    crystal_final=Crystal(reallat, l_atoms_itr)
    return crystal_final, en_final