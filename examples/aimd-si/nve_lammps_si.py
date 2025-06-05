import numpy as np
import matplotlib.pyplot as plt
import time
from lammps import lammps  # Added import for LAMMPS Python interface

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
from qtm.constants import RYDBERG
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.mpi.gspace import DistGSpace
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf
from qtm.force import force, force_ewald, force_local, force_nonloc
from qtm.MD import lammps_NVE  # Changed from NVE_MD to lammps_NVE

from qtm.io_utils.dft_printers import print_scf_status

import argparse

from qtm.config import qtmconfig
from qtm.logger import qtmlogger

initial_time = time.time()

# qtmconfig.fft_backend = "pyfftw"
qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
# Only G-space parallelization
# K-point and/or band parallelization along with G-space parallelization is currently broken.
n_kgrp=1
pwgrp_size=comm_world.size
dftcomm = DFTCommMod(comm_world, n_kgrp, pwgrp_size)

parser = argparse.ArgumentParser()
parser.add_argument("supercell_size", help="Side length of the supercell", type=int)

args = parser.parse_args()
supercell_size = args.supercell_size

alat = 10.2
# Lattice
reallat = RealLattice.from_alat(
    alat, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]  # Bohr
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

crystal_unit = Crystal(reallat, [si_atoms])
crystal_supercell = crystal_unit.gen_supercell([supercell_size] * 3)
r_alat_supercell = crystal_supercell.l_atoms[0].r_alat.T

if dftcomm.image_comm.rank == 0:
    print("the original coordinates are\n", r_alat_supercell, "\n")

crystal = crystal_supercell

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (1, 1, 1)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 7 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho_serial = GSpace(crystal.recilat, ecut_rho)

if dftcomm.n_pwgrp == dftcomm.image_comm.size:
    grho = grho_serial
else:
    grho = DistGSpace(comm_world, grho_serial)
gwfn = grho

#print("the type of grho is", type(grho))

numbnd = int(1.2 * (crystal.numel // 2))  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-4 * RYDBERG

## Smearing
steps = 5
dt = 20
mixing_beta = 0.3
max_t = steps * dt
T_init = 300
smear_typ = 'gauss'
e_temp = 1E-2 * RYDBERG
occ_typ = 'smear'

smear_print = 1 if smear_typ == 'gauss' else 0
occ_print = 1 if occ_typ == 'smear' else 0

if dftcomm.image_comm.rank == 0:
    print("nstep=", steps)
    print("\ndt=", dt)
    print("\necutwfc=", ecut_wfn)
    print("\nocc_typ=", occ_print)
    print("\nsmear_typ=", smear_print)
    print("\ne_temp=", e_temp)
    print("\nconv_thr=", conv_thr)
    print("\ndiago_thr_init=", diago_thr_init)
    print("\nmixing_beta=", mixing_beta)
    print("\nnumbnd=", numbnd)
    print("\ninitialtemp=", T_init)
    print(flush=True)

from time import perf_counter

initial_time = perf_counter()

# Replaced NVE_MD with lammps_NVE
out = lammps_NVE(dftcomm, crystal, max_t, dt, T_init, kpts, grho, gwfn, ecut_wfn,
                 numbnd, is_spin=False, is_noncolin=False,
                 symm_rho=False, rho_start=None, occ_typ='smear',
                 smear_typ='gauss', e_temp=1E-2 * RYDBERG,
                 conv_thr=conv_thr, diago_thr_init=diago_thr_init,
                 iter_printer=print_scf_status)

coords, time_MD, temperature, energy = out
au_to_ps=0.024188845
step_to_ps=dt*au_to_ps
time_MD*=step_to_ps

if dftcomm.image_comm.rank == 0:
    print("the coordinates are\n", coords, "\n")
    print("the time is\n", time_MD, "\n")
    print("the temperature is\n", temperature, "\n")
    print("the energy is\n", energy, "\n")

final_time = perf_counter()
print("The time taken is", final_time - initial_time)
print("Approx time per iteration", (final_time - initial_time) / steps)

## Plotting the Temperature
plt.plot(time_MD, temperature)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend([f"each time step represents {dt} atomic units"])
plt.savefig('Temperature_vs_Time_big_lammps.png')

## Plotting the Energy
plt.figure()
plt.plot(time_MD, energy)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Total Energy vs Time')
plt.legend([f"each time step represents {dt} atomic units"])
plt.savefig('Energy_vs_Time_big_lammps.png')

## Saving as txt files
np.savetxt("time_big_lammps.txt", time_MD)
np.savetxt("temperature_big_lammps.txt", temperature)
np.savetxt("energy_big_lammps.txt", energy)

print("SCF Routine has exited")
print(qtmlogger)