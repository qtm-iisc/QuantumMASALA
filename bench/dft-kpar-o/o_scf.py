from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG, ELECTRONVOLT
from quantum_masala.core import RealLattice, AtomBasis, Crystal
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KPoints

from quantum_masala.core import GSpace, GField
from quantum_masala.pseudo import rho_generate_atomic
from quantum_masala.dft import scf
from time import perf_counter

from quantum_masala.utils.dft_printers import (
    print_crystal_info, print_kpoints, print_gspc_info,
    print_scf_status
)

config.numkgrp = None  # Only k-point parallelization
pwcomm = config.pwcomm

start_time = perf_counter()

# Lattice
reallat = RealLattice.from_angstrom(
    alat=4.27163,
    a1=[4.27163000000000, 0.00000000000000, 0.00000000000000],
    a2=[0.00000000000000, 4.27682000000000, 0.00000000000000],
    a3=[1.80666882186211, 0.00000000000000, 4.19933056075744],
)

# Atom Basis
o_oncv = UPFv2Data.from_file('O', 'O_ONCV_PBE-1.2.upf')

o_atoms = AtomBasis.from_angstrom(
    'O', 15.999, o_oncv, reallat,
    [4.2710041627, 1.0692050000, 0.6159158133],
    [1.8072946592, 1.0692050000, 3.5834147474],
)

o1_atoms = AtomBasis.from_angstrom(
    'O1', 15.999, o_oncv, reallat,
    [2.1351891627, 3.2076150000, 0.6159158133],
    [3.9431096592, 3.2076150000, 3.5834147474],
)

crystal = Crystal(reallat, [o_atoms, o1_atoms])  # Represents the crystal
recilat = crystal.recilat

# k-points
mpgrid_shape = (26, 24, 24)
mpgrid_shift = (True, True, True)
kpts = KPoints.mpgrid(crystal, mpgrid_shape, mpgrid_shift)


# G-Space
ecut_wfn = 100 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal, ecut_rho)

# Other parameters
is_spin, is_noncolin = True, False
numbnd = int(max(1.2*(crystal.numel/2), crystal.numel/2 + 4))
l_start_mag = [0.2, -0.2]

rhoatomic = GField.zeros(grho, 1 + is_spin)
for typ, start_mag in zip(crystal.l_atoms, l_start_mag):
    spin_pol = [0.5 + start_mag/2, 0.5 - start_mag/2]
    rhoatomic += spin_pol * rho_generate_atomic(typ, grho)
rho_start = rhoatomic

occ = 'smear'
smear_typ = 'gauss'
e_temp = 0.01 * ELECTRONVOLT  # equals 0.0007349862 Ry

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG

print_crystal_info(crystal)
print_kpoints(kpts)
print_gspc_info(grho, grho)
pwcomm.world_comm.barrier()

out = scf(crystal=crystal, kpts=kpts, rho_start=rho_start, symm_rho=True,
          numbnd=numbnd, is_spin=is_spin, is_noncolin=is_noncolin,
          wfn_init=None,
          xc_params={'exch_name': 'gga_x_pbe', 'corr_name': 'gga_c_pbe'},
          occ=occ, smear_typ=smear_typ, e_temp=e_temp,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out

if pwcomm.world_rank == 0:
    if scf_converged:
        print("SCF Routine has achieved convergence")
    print("SCF Routine has exited")
    print(pw_logger)
    print(f"!! TOTAL RUNTIME: {perf_counter() - start_time:.2f} secs")
