from quantum_masala.constants import RYDBERG
from quantum_masala.core import (
    RealLattice, AtomBasis, Crystal, GSpace, KList
)
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.dft import scf
from quantum_masala.utils.dft_printers import (
    print_crystal_info, print_kpoints, print_gspc_info, print_scf_status
)

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
print_crystal_info(crystal)

# Generating k-points from a Monkhorst Pack grid
mpgrid_shape = (4, 4, 4)
mpgrid_shift = (True, True, True)
kpts = KList.mpgrid(crystal, mpgrid_shape, mpgrid_shift)
print_kpoints(kpts)

# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
gspc_rho = GSpace(crystal, ecut_rho)
gspc_wfn = gspc_rho
print_gspc_info(gspc_rho, gspc_wfn)

# -----Spin-unpolarized calculation-----
is_spin, is_noncolin = False, False
rho_start = rhoatomic  # Atomic density as starting charge density for SCF Iteration
numbnd = int(crystal.numel // 2)

out = scf(crystal, kpts, gspc_rho, gspc_wfn, numbnd, is_spin, is_noncolin,
          occ='fixed', iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out

if pwcomm.world_rank == 0:
    print(f"SCF Routine has {'NOT' if not scf_converged else ''} "
          f"achieved convergence")
    print()

print_bands(l_wfn_kgrp)

if pwcomm.world_rank == 0:
    print("SCF Routine has exited")
    print(pw_logger)
