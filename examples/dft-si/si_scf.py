import numpy as np
from qtm.constants import RYDBERG, ELECTRONVOLT
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

from mpi4py.MPI import COMM_WORLD
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, 1)

# Lattice
reallat = RealLattice.from_alat(alat=10.2,  # Bohr
                                a1=[-0.5,  0. ,  0.5],
                                a2=[ 0. ,  0.5,  0.5],
                                a3=[-0.5,  0.5,  0. ])

# Atom Basis
si_oncv = UPFv2Data.from_file('Si_ONCV_PBE-1.2.upf')
si_atoms = BasisAtoms('si', si_oncv, 28.086, reallat, np.array(
    [[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]
).T)

crystal = Crystal(reallat, [si_atoms, ])  # Represents the crystal


# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 4, 4)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift, False, False)

# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

numbnd = 30  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG

out = scf(dftcomm, crystal, kpts, grho, gwfn,
          numbnd, is_spin=False, is_noncolin=False,
          symm_rho=False, rho_start=None, occ_typ='fixed',
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out

print("SCF Routine has exited")
print(qtmlogger)
