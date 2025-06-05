from __future__ import annotations
import gc
import tracemalloc
import numpy as numpy
from lammps import lammps

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

    class WfnInit(Protocol):
        def __call__(self, ik: int, kswfn: list[KSWfn]) -> None:
            ...
else:
    IterPrinter = 'IterPrinter'
    WfnInit = 'WfnInit'

def write_lammps_data(filename, l_atoms, primvec, coords_cart_all, bohr_to_angstrom):
    """Write a LAMMPS data file with triclinic box and fractional coordinates."""
    # Compute QR decomposition
    ##Primvec have vectors as columns
    A,B,C= primvec.T
    a_x= np.linalg.norm(A)
    b_x = np.dot(A, B) / a_x
    b_y=np.sqrt(np.dot(B, B) - b_x**2)
    c_x= np.dot(A, C) / a_x
    c_y= (np.dot(B, C) - b_x*c_x) / b_y
    c_z= np.sqrt(np.dot(C, C) - c_x**2 - c_y**2)
    new_primvec= np.array([[a_x, b_x, c_x], [0.0, b_y, c_y], [0.0, 0.0, c_z]])
    
    '''Q, R = np.linalg.qr(primvec.T)
    
    # Ensure R has positive diagonal elements
    D = np.diag(np.sign(np.diag(R)))  # Diagonal matrix with signs of R's diagonal
    Q = Q @ D  # Adjust Q
    R = D @ R  # Adjust R to have positive diagonal
    L = R.T   # Convert to Angstroms, ensuring positive lengths
    
    # Verify box lengths
    if np.any(np.diag(L) <= 0):
        raise ValueError(f"Invalid box lengths: {np.diag(L)}. All must be positive.")'''
    
    # Compute fractional coordinates in the new coordinate system
    V= np.dot(C, np.cross(A, B))
    cross_matrix=np.array([np.cross(A,B), np.cross(B,C), np.cross(C,A)])
    conv= new_primvec@np.linalg.inv(primvec)
    s = coords_cart_all
    ##Reduce s to 0, if it is very close to 0.
    s[np.abs(s) < 1e-10] = 0.0
    # Wrap to [0,1]
    ##Write in Bohr by multiplying with

    '''coords_cart_orig= np.dot(coords__all, primvec.T)
    coords_cart_all= np.dot(s, new_primvec.T)

    print(f"Coordinates in original system: {coords_cart_orig}")
    print(f"Coordinates in new system: {coords_cart_all}")'''


    # Assign atom types based on species
    tot_num = sum(sp.numatoms for sp in l_atoms)
    atom_types = np.concatenate([np.full(sp.numatoms, i+1) for i, sp in enumerate(l_atoms)])

    # Write the data file
    with open(filename, 'w') as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{tot_num} atoms\n")
        f.write(f"{len(l_atoms)} atom types\n")
        f.write(f"-30.6 30.6 xlo xhi\n")
        f.write(f"-30.6 30.6 ylo yhi\n")
        f.write(f"-30.6 30.6 zlo zhi\n")
        #f.write(f"{b_x:.6f} {c_x:.6f} {c_y:.6f} xy xz yz\n\n")
        #f.write(f"{primvec[0,0]:.6f} {primvec[0,1]:.6f} {primvec[0,2]:.6f} avec\n\n")
        #f.write(f"{primvec[1,0]:.6f} {primvec[1,1]:.6f} {primvec[1,2]:.6f} bvec\n\n")
        #f.write(f"{primvec[2,0]:.6f} {primvec[2,1]:.6f} {primvec[2,2]:.6f} cvec\n\n")
        #f.write(f"0 0 0 abc origin\n\n")
        f.write("Masses\n\n")
        for i, sp in enumerate(l_atoms):
            f.write(f"{i+1} {sp.mass}\n")
        f.write("\nAtoms\n\n")
        for i in range(tot_num):
            f.write(f"{i+1} {atom_types[i]} {s[i,0]:.6f} {s[i,1]:.6f} {s[i,2]:.6f}\n")
    
    # Optional: Print for debugging
    print(f"Box bounds: xlo xhi = 0.0 {a_x}, ylo yhi = 0.0 {b_y}, zlo zhi = 0.0 {c_z}")
    print(f"Tilt factors: xy = {b_x}, xz = {c_x}, yz = {c_y}")

def lammps_NVE(dftcomm,
           crystal: Crystal,
           max_t: float,
           dt: float,
           T_init: float,
           kpts: KList,
           grho: GSpace,
           gwfn: GSpace,
           ecut_wfn: float,
           numbnd: int,
           is_spin: bool,
           is_noncolin: bool,
           symm_rho: bool = True,
           rho_start: 'FieldGType | tuple[float, ...] | None' = None,
           wfn_init: 'WfnInit | None' = None,
           libxc_func: 'tuple[str, str] | None' = None,
           occ_typ: 'Literal["fixed", "smear"]' = 'smear',
           smear_typ: 'Literal["gauss", "fd", "mv"]' = 'gauss',
           e_temp: float = 1E-3,
           conv_thr: float = 1E-6 * RYDBERG,
           maxiter: int = 100,
           diago_thr_init: float = 1E-2 * RYDBERG,
           iter_printer: 'IterPrinter | None' = None,
           mix_beta: float = 0.7,
           mix_dim: int = 8,
           dftconfig: 'DFTConfig | None' = None,
           ret_vxc: bool = False,
           gamma_only: bool = False):
    # Extract crystal properties
    l_atoms = crystal.l_atoms
    tot_num = np.sum([sp.numatoms for sp in l_atoms])
    num_in_types=np.array([sp.numatoms for sp in l_atoms])
    ppdat_cryst=np.array([sp.ppdata for sp in l_atoms])
    label_cryst=np.array([sp.label for sp in l_atoms])
    mass_cryst=np.array([sp.mass for sp in l_atoms])*M_NUC_HART
    mass_all=np.repeat([sp.mass for sp in l_atoms], [sp.numatoms for sp in l_atoms])*M_NUC_HART
    num_typ = len(l_atoms)
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1).T 
    coords_cryst_all = np.concatenate([sp.r_cryst for sp in l_atoms], axis=1).T
    reallat=crystal.reallat # (num_atoms, 3)

    # Unit conversion factors
    bohr_to_angstrom = 0.529177
    hartree_to_ev = 27.211386
    time_au_to_ps = 2.418884326505e-5
    force_conv = hartree_to_ev / bohr_to_angstrom  # Hartree/bohr to eV/Angstrom
    vel_conv = bohr_to_angstrom / time_au_to_ps  # a.u. to Angstrom/ps



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

    lmp = lammps()
    print("Initialized lammps")
    

    write_lammps_data("data.lmp", l_atoms, reallat.primvec, coords_cart_all, bohr_to_angstrom)

    # Initialize LAMMPS
    print("initialized lammps")
    lmp.command("units metal")
    print("Initialized units", flush=True)
    lmp.command("atom_style atomic")
    print("Initialized atom style", flush=True)
    lmp.command("read_data data.lmp")
    print("Read data file", flush=True)
    lmp.command("pair_style none")  # Forces computed externally
    print("Set pair style", flush=True)
    lmp.command("fix 1 all nve")
    print("Set NVE fix", flush=True)
    lmp.command(f"timestep {dt * time_au_to_ps}")
    print("Set timestep", flush=True)

    #Set initial velocities directly in LAMMPS
    lmp.command(f"velocity all create {T_init} 12345 dist gaussian")
    print("Set initial velocities")
    lmp.command("velocity all zero linear") 
    print(f"Initialized velocities in LAMMPS with temperature: {T_init} K")

    # Extract velocities to verify (optional)
    v = lmp.numpy.extract_atom("v")
    v_tot=dftcomm.image_comm.allgather(v)
    v_tot= np.concatenate(v_tot, axis=0)
    v_tot= np.reshape(v_tot, (tot_num, 3))
    print("Velocities extracted from LAMMPS (Angstrom/ps", v_tot, "in processor",flush=True)  # Print first 5 for brevity
    v=v_tot
    # Convert to atomic units for consistency with your code (optional)
    vel_lmp = v  # Already in Angstrom/ps from LAMMPS
    vel = vel_lmp / vel_conv  # Convert back to a.u. if needed later
    ke = 0.5 * np.sum(mass_all *vel.T**2)  # Kinetic energy in Hartree
    T = 2 * ke / (3 * tot_num * BOLTZMANN_HART)  # Temperature in K
    print(f"Actual temperature after initialization: {T:.2f} K")
    

    # Arrays to store time, temperature, and energy
    time_steps = int(max_t / dt)
    time_array = np.empty(time_steps)
    temperature_array = np.empty(time_steps)
    energy_array = np.empty(time_steps)
    print("coords cart all from the coordinates", coords_cart_all, flush=True)

    # MD loop
    time = 0
    rho_md = rho_start
    for step in range(time_steps):
        # Get current positions from LAMMPS (in Angstroms)
        x = lmp.numpy.extract_atom("x")
        print("x from LAMMPS", x,flush=True)
        x_tot=dftcomm.image_comm.allgather(x)
        x_tot= np.concatenate(x_tot, axis=0)
        x_tot= np.reshape(x_tot, (tot_num, 3))
        x=x_tot
          # Print first 5 for brevity
        coords_cart_all = x

        # Convert to bohr
        print(f"Step {step}: Coordinates from LAMMPS in bohr alat units: {coords_cart_all}", flush=True)
        #coords_cart_all = reallat.alat*coords_cart_all  # Convert
        print(f"Step {step}: Coordinates from LAMMPS: {coords_cart_all}", flush=True)

        # Compute energy and forces using DFT
        with dftcomm.image_comm as comm: en, force_coord, rho_md = compute_en_force(dftcomm,coords_cart_all, rho_md)
        # Convert forces to LAMMPS units (eV/Angstrom)
        f_lmp = force_coord * force_conv * RYDBERG  # en and force_coord in Hartree, convert to eV/Angstrom
        ID= lmp.numpy.extract_atom("id")-1
        print("ID in this processor", ID, len(ID), flush=True)
        if len(ID)>0: 
            f = lmp.numpy.extract_atom("f")
            print("f in this processor", f, flush=True)
            f_ext=f_lmp[ID]
            print("f_ext in this processor", f_ext, flush=True) 
            f[:]=f_ext
        # Run one time step in LAMMPS
        f_new = lmp.numpy.extract_atom("f")
        print("f_new in this processor", f_new, flush=True)
        lmp.command("run 1")

        # Compute kinetic energy and temperature
        v = lmp.numpy.extract_atom("v")
        v_tot=dftcomm.image_comm.allgather(v)
        v_tot= np.concatenate(v_tot, axis=0)
        v_tot= np.reshape(v_tot, (tot_num, 3))
        print("v from LAMMPS", v_tot, flush=True)
        v=v_tot
        vel = v / vel_conv  # Convert back to a.u.
        ke = 0.5 * np.sum(mass_all * vel.T**2)  # in Hartree
        T = 2 * ke / (3 * tot_num * BOLTZMANN_HART)  # in K
        en_total = en + ke  # Total energy in Hartree

        # Store and print results
        time_array[step] = time
        temperature_array[step] = T
        energy_array[step] = en_total
        print(f"Time: {time:.6f} a.u., Total Energy: {en_total * hartree_to_ev:.6f} eV, "
            f"Potential Energy: {en * hartree_to_ev:.6f} eV, "
            f"Kinetic Energy: {ke * hartree_to_ev:.6f} eV, Temperature: {T:.2f} K")

        time += dt

    # Broadcast results to all processes
    comm.Bcast(time_array, root=0)
    comm.Bcast(temperature_array, root=0)
    comm.Bcast(energy_array, root=0)
    x = lmp.numpy.extract_atom("x")
    x_tot=dftcomm.image_comm.allgather(x)
    x_tot= np.concatenate(x_tot, axis=0)
    x_tot= np.reshape(x_tot, (tot_num, 3))
    x=x_tot
    coords_cart_all = x / bohr_to_angstrom

    # Clean up LAMMPS instance
    lmp.close()

    return coords_cart_all, time_array, temperature_array, energy_array