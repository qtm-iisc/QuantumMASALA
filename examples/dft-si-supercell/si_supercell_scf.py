from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG
from quantum_masala.core import RealLattice, AtomBasis
from quantum_masala.core import Crystal, crystal_gen_supercell
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KList

from quantum_masala.core import GSpace
from quantum_masala.dft import scf
from time import perf_counter

from quantum_masala.utils.dft_printers import (
    print_crystal_info, print_kpoints, print_gspc_info,
    print_scf_status, print_bands
)

config.numkgrp = 1  # Only 1 kgrp; Band parallelization
config.use_gpu = False
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
supercell_dim = (3, 3, 3)
print(f"Generating a {supercell_dim} supercell")
crystal = crystal_gen_supercell(crystal, supercell_dim)

reallat = crystal.reallat
recilat = crystal.recilat

kpts = KList.gamma(crystal)

ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
gspc_rho = GSpace(crystal, ecut_rho)
gspc_wfn = gspc_rho

is_spin, is_noncolin = False, False
numbnd = crystal.numel // 2

print_crystal_info(crystal)
print_kpoints(kpts)
print_gspc_info(gspc_rho, gspc_wfn)
pwcomm.world_comm.barrier()

out = scf(crystal, kpts, gspc_rho, gspc_wfn, numbnd, is_spin, is_noncolin,
          occ='fixed', conv_thr=1E-6 * RYDBERG, diago_thr_init=1E-2 * RYDBERG,
          iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out

if pwcomm.world_rank == 0:
    print(f"SCF Routine has {'NOT' if not scf_converged else ''} "
          f"achieved convergence")
    print()

print_bands(l_wfn_kgrp)

if pwcomm.world_rank == 0:
    print("SCF Routine has exited")
    print(pw_logger)
    print("! Total Run Time: ", perf_counter() - start_time)
