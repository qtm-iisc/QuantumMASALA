from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG, ELECTRONVOLT
from quantum_masala.core import RealLattice, AtomBasis, Crystal
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KList

from quantum_masala.core import GSpace
from quantum_masala.pseudo import rho_generate_atomic
from quantum_masala.dft import scf

from quantum_masala.utils.dft_printers import (
    print_crystal_info, print_kpoints, print_gspc_info,
    print_scf_status, print_bands
)

pwcomm = config.pwcomm


# Lattice
reallat = RealLattice.from_alat(alat=5.1070,  # Bohr
                                a1=[ 0.5,  0.5,  0.5],
                                a2=[-0.5,  0.5,  0.5],
                                a3=[-0.5, -0.5,  0.5])

# Atom Basis
fe_oncv = UPFv2Data.from_file('fe', 'Fe_ONCV_PBE-1.2.upf')
fe_atoms = AtomBasis.from_cart('fe', 55.487, fe_oncv, reallat, [0., 0., 0.])

crystal = Crystal(reallat, [fe_atoms, ])  # Represents the crystal
recilat = crystal.recilat

print_crystal_info(crystal)

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (8, 8, 8)
mpgrid_shift = (True, True, True)
kpts = KPoints.mpgrid(crystal, mpgrid_shape, mpgrid_shift)

print_kpoints(kpts)

# -----Setting up G-Space of calculation-----
ecut_wfn = 40 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal, ecut_rho)
gwfn = grho

print_gspc_info(grho, gwfn)

# Initializing starting charge density from superposition of atomic charges
rhoatomic = rho_generate_atomic(crystal.l_atoms[0], grho)

# -----Spin-polarized (collinear) calculation-----
is_spin, is_noncolin = True, False
# Starting with asymmetric spin distribution else convergence may yield only
# non-magnetized states
rho_start = [0.55, 0.45] * rhoatomic
numbnd = 12  # Ensure adequate # of bands if system is not an insulator

occ = 'smear'
smear_typ = 'gauss'
e_temp = 1E-2 * RYDBERG

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG


out = scf(crystal=crystal, kpts=kpts, rho_start=rho_start, symm_rho=True,
          numbnd=numbnd, is_spin=is_spin, is_noncolin=is_noncolin,
          wfn_init=None,
          xc_params={'exch_name': 'gga_x_pbe', 'corr_name': 'gga_c_pbe'},
          occ=occ, smear_typ=smear_typ, e_temp=e_temp,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out

if pwcomm.world_rank == 0:
    print(f"SCF Routine has {'NOT' if not scf_converged else ''} "
          f"achieved convergence")
    print()

print_bands(l_wfn_kgrp)

pwcomm.world_comm.barrier()
if pwcomm.world_rank == 0:
    print("SCF Routine has exited")
    print(pw_logger)

