from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG, ELECTRONVOLT
from quantum_masala.core import RealLattice, AtomBasis, Crystal
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KList

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

crystal = Crystal(reallat, [si_atoms, ])  # Represents the crystal
recilat = crystal.recilat

print_crystal_info(crystal)

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 4, 4)
mpgrid_shift = (True, True, True)
kpts = KList.mpgrid(crystal, mpgrid_shape, mpgrid_shift)

# Alternatively, k-points can be set from input list
# kpts = KList.from_tpiba(crystal,
#     [(-0.125,  0.125,  0.125), 0.0625],
#     [(-0.375,  0.375, -0.125), 0.1875],
#     [( 0.375, -0.375,  0.625), 0.1875],
#     [( 0.125, -0.125,  0.375), 0.1875],
#     [(-0.125,  0.625,  0.125), 0.1875],
#     [( 0.625, -0.125,  0.875), 0.3750],
#     [( 0.375,  0.125,  0.625), 0.3750],
#     [(-0.125, -0.875,  0.125), 0.1875],
#     [(-0.375,  0.375,  0.375), 0.0625],
#     [( 0.375, -0.375,  1.125), 0.1875],
# )

print_kpoints(kpts)

# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal, ecut_rho)
gwfn = grho

print_gspc_info(grho, gwfn)

# Initializing starting charge density from superposition of atomic charges
rhoatomic = rho_generate_atomic(crystal.l_atoms[0], grho)

# -----Spin-unpolarized calculation-----
is_spin, is_noncolin = False, False
rho_start = rhoatomic  # Atomic density as starting charge density for SCF Iteration
numbnd = int(crystal.numel // 2)

occ = 'fixed'

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG


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
