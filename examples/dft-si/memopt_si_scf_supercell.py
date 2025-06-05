"""
This example file demonstrates the usage of G-space parallelization in QuantumMASALA.

The code performs a self-consistent field (SCF) calculation for a silicon supercell.

The main steps of the code are as follows:
1. Import necessary modules and libraries.
2. Set up the communication world for parallelization.
3. Define the lattice and atom basis for the crystal.
4. Generate the supercell based on the specified size.
5. Generate k-points using a Monkhorst Pack grid.
6. Set up the G-Space for the calculation.
7. Perform the SCF calculation using the specified parameters.
8. Print the SCF convergence status and results.

Example usage:
python si_scf_supercell.py <supercell_size>

Parameters:
- supercell_size: The size of the supercell in each dimension.

Output:
- SCF convergence status and results.

"""
import numpy as np
from qtm.constants import RYDBERG
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.mpi.gspace import DistGSpace
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_scf_status

import argparse

from qtm import qtmconfig
from qtm.logger import qtmlogger

# qtmconfig.fft_backend = "pyfftw"
qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)


# Memory tracing
# =============================================================================
import tracemalloc
# # Flag to determine if we select NumPy domain
# use_np_domain = True
# # Start to trace memory
tracemalloc.start()
# =============================================================================

# Only G-space parallelization
# K-point and/or band parallelization along with G-space parallelization is currently broken.
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)

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
    np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]).T,
)

crystal = Crystal(reallat, [si_atoms])  # Represents the crystal

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
grho_serial = GSpace(crystal.recilat, ecut_rho)

# If G-space parallelization is not required, use the serial G-space object
if dftcomm.n_pwgrp == dftcomm.image_comm.size:  
    grho = grho_serial
else:
    grho = DistGSpace(comm_world, grho_serial)
gwfn = grho

numbnd = crystal.numel // 2  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

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

scf_converged, rho, l_wfn_kgrp, en = out

if comm_world.rank == 0:
    print("SCF Routine has exited")
    print(qtmlogger)

# =============================================================================

current, peak = tracemalloc.get_traced_memory()
print("Communication world size: ", comm_world.size)
# print("G-space parallelization: ", dftcomm.pwgrp_intra.size)
print(f"Peak memory usage: {peak / 10**6}MB; Current memory usage: {current / 10**6}MB")

if comm_world.rank == 0:
    with open(f"memopt_si_scf_supercell_{supercell_size}_mpi_{comm_world.size}.txt", "w") as f:
        f.write(f"Supercell size: {supercell_size}\n")
        f.write(f"Communication world size: {comm_world.size}\n")
        # f.write(f"G-space parallelization: {dftcomm.pwgrp_intra.size}\n")
        f.write(f"Peak memory usage: {peak / 10**6}MB; Current memory usage: {current / 10**6}MB\n")
        f.write(f"Result: {supercell_size}, {comm_world.size}, {peak / 10**6}, {current / 10**6}")
tracemalloc.stop()

# =============================================================================
