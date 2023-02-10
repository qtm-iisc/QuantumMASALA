from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG, ELECTRONVOLT
from quantum_masala.core import RealLattice, AtomBasis
from quantum_masala.core import Crystal, crystal_gen_supercell
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KPoints

from quantum_masala.core import GSpace
from quantum_masala.pseudo import rho_generate_atomic
from quantum_masala.dft import scf
from time import perf_counter

from quantum_masala.utils.dft_printers import (
    print_crystal_info, print_kpoints, print_gspc_info,
    print_scf_status, print_bands
)

pwcomm = config.pwcomm

start_time = perf_counter()


# Lattice
reallat = RealLattice.from_alat(alat=10.2,  # Bohr
                                a1=[-0.5,  0.0,  0.5],
                                a2=[ 0.0,  0.5,  0.5],
                                a3=[-0.5,  0.5,  0.0])

# Atom Basis
si_oncv = UPFv2Data.from_file('Si', 'Si_ONCV_PBE-1.2.upf')
si_atoms = AtomBasis.from_alat('Si', 28.085, si_oncv, reallat,
                               [0., 0., 0.], [0.25, 0.25, 0.25])

crystal = Crystal(reallat, [si_atoms, ])
# Generating supercell
supercell_dim = (5, 5, 5)
print(f"Generating a {supercell_dim} supercell")
crystal = crystal_gen_supercell(crystal, supercell_dim)

reallat = crystal.reallat
recilat = crystal.recilat

kpts = KPoints.gamma(crystal)

ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal, ecut_rho)

rhoatomic = rho_generate_atomic(crystal.l_atoms[0], grho)

is_spin, is_noncolin = False, False
rho_start = rhoatomic
numbnd = crystal.numel // 2

occ = 'fixed'

conv_thr = 1E-6 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG

print_crystal_info(crystal)
pwcomm.world_comm.barrier()

print_kpoints(kpts)
pwcomm.world_comm.barrier()

print_gspc_info(grho, grho)

out = scf(crystal=crystal, kpts=kpts, rho_start=rho_start, symm_rho=True,
          numbnd=numbnd, is_spin=is_spin, is_noncolin=is_noncolin,
          wfn_init=None,
          xc_params={'exch_name': 'gga_x_pbe', 'corr_name': 'gga_c_pbe'},
          occ=occ, conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status
          )

scf_converged, rho, l_wfn_kgrp, en = out

if pwcomm.world_rank == 0:
    print(f"SCF Routine has {'NOT' if not scf_converged else ''} "
          f"achieved convergence")
    print()

print_bands(l_wfn_kgrp)


if pwcomm.world_rank == 0:
    print("SCF Routine has exited")
    print(pw_logger)
