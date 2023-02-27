from time import perf_counter

from quantum_masala.constants import RYDBERG
from quantum_masala.core import (
    RealLattice, AtomBasis, Crystal, GSpace, KList
)
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.dft import scf
from quantum_masala.utils.dft_printers import (
    print_crystal_info, print_kpoints, print_gspc_info, print_scf_status
)
from quantum_masala import config, pw_logger

start_time = perf_counter()

pwcomm = config.pwcomm

reallat = RealLattice.from_angstrom(
    alat=2.83351,
    a1=[2.83351000000000, 0.00000000000000, 0.00000000000000],
    a2=[0.00000000000000, 2.83351000000000, 0.00000000000000],
    a3=[0.00000000000000, 0.00000000000000, 2.83351000000000],
)

fe_oncv = UPFv2Data.from_file('Fe', 'Fe_ONCV_PBE-1.2.upf')
fe_atoms = AtomBasis.from_angstrom('Fe', 55.845, fe_oncv, reallat,
                                   [0.7083775000, 0.7083775000, 0.7083775000],
                                   [2.1251325000, 2.1251325000, 2.1251325000])

crystal = Crystal(reallat, [fe_atoms])

kpts = KList.mpgrid(crystal, (10, 10, 10), (True, True, True))

ecut_wfn = 100 * RYDBERG
ecut_rho = 4 * ecut_wfn
gspc_rho = GSpace(crystal, ecut_rho)
gspc_wfn = gspc_rho

is_spin, is_noncolin = False, False
numbnd = round(1.2 * (crystal.numel // 2))

print_crystal_info(crystal)
print_kpoints(kpts)
print_gspc_info(gspc_rho, gspc_wfn)
pwcomm.world_comm.barrier()


out = scf(crystal, kpts, gspc_rho, gspc_wfn, numbnd, is_spin, is_noncolin,
          occ='smear', smear_typ='gauss', e_temp=0.0007349862 * RYDBERG,
          conv_thr=1E-8 * RYDBERG, diago_thr_init=1E-2,
          iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out

pwcomm.world_comm.barrier()
if pwcomm.world_rank == 0:
    print("SCF Routine has exited")
    print(pw_logger)
    print("! Total Run Time: ", perf_counter() - start_time)
