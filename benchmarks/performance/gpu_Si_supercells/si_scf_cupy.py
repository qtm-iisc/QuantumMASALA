import numpy as np
from qtm.constants import RYDBERG
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf
import cupy

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger

import argparse

# qtmconfig.fft_backend = "mkl_fft"

from mpi4py.MPI import COMM_WORLD

comm_world = QTMComm(COMM_WORLD)

# Only k-pt parallelization:
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)
# Only band parallelization:
# dftcomm = DFTCommMod(comm_world, 1, 1)

# Lattice
reallat = RealLattice.from_alat(
    alat=10.2, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]  # Bohr
)

# Atom Basis
si_oncv = UPFv2Data.from_file("Si_ONCV_PBE-1.2.upf")
si_atoms = BasisAtoms(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0.875, 0.875, 0.875]]).T,
)
si_atoms_2 = BasisAtoms(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0.125, 0.125, 0.125]]).T,
)

crystal = Crystal(reallat, [si_atoms, si_atoms_2])  # Represents the crystal

parser = argparse.ArgumentParser()
parser.add_argument("supercell_size", help="Side length of the supercell", type=int)
args = parser.parse_args()
supercell_size = args.supercell_size

crystal = crystal.gen_supercell([supercell_size] * 3)


# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (1, 1, 1)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

numbnd = crystal.numel // 2  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-5 * RYDBERG

cupy.cuda.runtime.profilerStart()
out = scf(
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin=False,
    is_noncolin=False,
    symm_rho=True,
    rho_start=None,
    occ_typ="fixed",
    conv_thr=conv_thr,
    diago_thr_init=diago_thr_init,
    iter_printer=print_scf_status,
)
cupy.cuda.runtime.profilerStop()
scf_converged, rho, l_wfn_kgrp, en = out

if comm_world.rank == 0:
    print("SCF Routine has exited")
    print(qtmlogger)