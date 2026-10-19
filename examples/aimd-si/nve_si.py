import numpy as np
import matplotlib.pyplot as plt
import time
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
from qtm.pot.force import force, force_ewald, force_local, force_nonloc
from qtm.MD import NVE_MD, RDist

from qtm.io_utils.dft_printers import print_scf_status

import argparse

from qtm.config import qtmconfig
from qtm.logger import qtmlogger

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
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)

parser = argparse.ArgumentParser()
parser.add_argument("supercell_size", help="Side length of the supercell", type=int)

args = parser.parse_args()
supercell_size = args.supercell_size

alat=10.2612
# Lattice
reallat = RealLattice.from_alat(
    alat, a1=[1, 0, 0], a2=[0, 1, 0], a3=[0, 0, 1]  # Bohr
)

# Atom Basis
si_oncv = UPFv2Data.from_file("Si_ONCV_PBE-1.2.upf")

si_atoms = BasisAtoms.from_alat(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0,0,0],[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]]),
)

crystal_unit = Crystal(reallat, [si_atoms])
crystal_supercell=crystal_unit.gen_supercell([supercell_size] * 3)
r_alat_supercell=crystal_supercell.l_atoms[0].r_alat.T
r_cart_supercell=crystal_supercell.l_atoms[0].r_cart.T

if dftcomm.image_comm.rank==0:
    print("the original coordinates are\n", r_alat_supercell, "\n")
    print("the original coordinates in cartesian are\n", r_cart_supercell, "\n")

crystal=crystal_supercell  # Represents the crystal

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (1, 1, 1)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
ecut_rho = 4 *ecut_wfn
grho_serial = GSpace(crystal.recilat, ecut_rho)

# If G-space parallelization is not required, use the serial G-space object
if dftcomm.n_pwgrp == dftcomm.image_comm.size:
    grho = grho_serial
else:
    grho = DistGSpace(comm_world, grho_serial)
gwfn = grho

numbnd = int(1.2*(crystal.numel // 2)) # Ensure adequate # of bands if system is not an insulator
conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-4 * RYDBERG

##Smearing
steps=50
dt=30
mixing_beta=0.3
max_t=steps*dt
T_init=300
smear_typ='gauss'
e_temp=1E-2 * RYDBERG
occ_typ='smear'
vel_init=np.array([
    [-2.34750948e-04,  1.82331329e-04, -6.19270033e-05],
    [ 3.63748441e-04, -2.87815496e-04,  5.40517365e-05],
    [-5.05165158e-04,  1.78049563e-04,  9.38352961e-05],
    [ 5.12264684e-04,  7.17466474e-04,  3.03161264e-04],
    [ 4.53145733e-04, -1.85746395e-04, -2.39916694e-04],
    [ 9.06840609e-05, -3.42811524e-05, -1.97946061e-04],
    [ 4.13981671e-04, -3.87588446e-04, -6.63613668e-04],
    [ 7.45518294e-05,  1.18854878e-04,  6.71739183e-05],
    [ 1.38684327e-04, -3.22553183e-04,  3.11349468e-04],
    [-5.10550434e-04, -1.91596313e-04,  2.25387106e-04],
    [-1.99613229e-04, -1.13174222e-04,  1.87212996e-04],
    [-6.22400908e-04, -9.97806001e-05,  3.28728913e-04],
    [ 4.40218330e-04, -4.66622249e-05,  1.24021589e-04],
    [-7.05563591e-04, -3.56000547e-04, -1.50592590e-04],
    [-2.72046097e-04,  1.24704568e-04,  1.64648673e-04],
    [ 5.82564772e-05,  3.09413316e-04,  1.01551189e-04],
    [-4.62730546e-04, -9.51864670e-06, -4.84343084e-04],
    [-8.76044635e-05,  2.50984893e-04, -1.57079885e-04],
    [-2.23834220e-04,  2.87483039e-04,  1.53726345e-04],
    [-1.38918835e-04,  1.45132682e-04, -5.63358807e-05],
    [ 3.12550551e-04, -1.50889428e-04,  7.23809556e-04],
    [ 1.49375054e-04,  3.86128059e-05, -9.43399260e-05],
    [-2.18101558e-04,  8.37867603e-05,  2.38038500e-05],
    [-6.91348733e-05,  3.09844747e-04, -1.14098488e-04],
    [-1.48930410e-04, -2.64001790e-04,  4.87608181e-06],
    [ 2.22984671e-04, -4.87332459e-05, -1.03114184e-05],
    [-7.05569218e-05, -3.12384274e-04, -3.70256479e-04],
    [-3.17962126e-04,  2.23999235e-04, -2.55229061e-04],
    [-1.65387228e-04, -2.50323515e-04,  1.45774723e-04],
    [ 3.85175297e-04, -5.53088282e-04,  1.92647473e-04],
    [-3.90103159e-04, -1.14189081e-04,  4.09097102e-04],
    [ 1.73325355e-04,  3.82902633e-04, -1.23066420e-04],
    [ 2.90928947e-04,  4.70735841e-05, -1.01049527e-04],
    [-3.54037296e-04, -3.07962101e-05,  1.70518996e-04],
    [ 7.18929590e-05,  6.35114546e-05,  2.40382615e-05],
    [ 1.64905462e-04, -1.86869209e-04, -4.65828002e-06],
    [-4.06905765e-05, -2.40808994e-04,  1.69176475e-04],
    [ 1.42299363e-05,  3.22396862e-04,  1.51752238e-04],
    [ 8.88697946e-05, -1.73059268e-04, -1.89686436e-04],
    [ 5.73104040e-05,  4.14151583e-05,  2.90122594e-04],
    [-1.59947470e-04, -9.05486707e-05,  2.11963845e-04],
    [-2.94399937e-04,  4.81312460e-04, -8.03242933e-05],
    [ 2.50693908e-04,  3.28130349e-04,  2.14775050e-04],
    [ 4.54556817e-04,  1.74207353e-04,  1.23843127e-04],
    [ 4.77101969e-04, -8.28056243e-05,  2.98566868e-04],
    [-9.45586515e-05,  5.12531225e-04, -2.66466457e-04],
    [ 6.15684655e-05,  9.02169311e-05, -1.08419484e-04],
    [ 5.83968394e-05, -5.09424214e-04, -1.96786637e-04],
    [-6.68246906e-05, -8.85716697e-05,  6.45388956e-04],
    [ 2.48480487e-04, -2.34578916e-04, -2.23260684e-04],
    [ 2.14638409e-05,  1.33137599e-04,  2.22307171e-05],
    [-1.01534449e-04, -2.63264330e-04,  1.81070813e-04],
    [ 2.92445000e-04,  1.97821542e-04, -4.17701643e-04],
    [ 4.30100938e-04, -2.90598277e-05,  9.27063261e-05],
    [-3.62600555e-04, -3.15790333e-04,  3.30797634e-04],
    [ 2.51765726e-04,  1.30565729e-04, -1.02216739e-04],
    [-2.12512145e-04,  1.89058704e-04, -6.93907977e-05],
    [-1.78396193e-06, -2.91304174e-04,  2.85471540e-04],
    [-5.53589616e-05, -2.92804105e-04, -2.70111523e-04],
    [ 2.46521022e-04, -1.35057704e-04,  1.38470036e-04],
    [-3.38005511e-04, -6.97214539e-05,  3.74961461e-04],
    [ 7.66632343e-05, -5.35226323e-04, -6.05655092e-04],
    [ 4.53362845e-05,  4.39629074e-04, -3.06132268e-04],
    [-7.09185577e-05, -2.52966849e-04, -2.69036977e-04]
])


smear_print=1 if smear_typ=='gauss' else 0
occ_print=1 if occ_typ=='smear' else 0

if dftcomm.image_comm.rank==0: 
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

rmax=np.array([0.5, 0.5, 0.5])
r, g_r=RDist(crystal, r_cart_supercell, rmax, 1000)
plt.figure()
plt.plot(r, g_r)
plt.xlabel('Distance (Bohr)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('RDF_initial.png')
# -----Running SCF calculation-----

from time import perf_counter

initial_time=perf_counter()

out = NVE_MD(dftcomm, crystal, max_t, dt, T_init, vel_init, kpts, grho, gwfn, ecut_wfn,
          numbnd, is_spin=False, is_noncolin=False,
          symm_rho=False, rho_start=None, occ_typ='smear',
            smear_typ='gauss', e_temp=1E-2 * RYDBERG,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

coords, time_MD, temperature, energy, msd, vacf, vel, RDF_data= out


au_to_ps=0.0241888426*1e-3
# Convert time to picosecond
time_MD*=dt*au_to_ps

r, g_r=RDist(crystal, coords, rmax, 1000)
plt.figure()
plt.plot(r, g_r)
plt.xlabel('Distance (Bohr)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('RDF_final.png')
# -----Running SCF calculation-----


final_time=perf_counter()
print("The time taken is", final_time-initial_time)
print("Approx time per iteration", (final_time-initial_time)/steps)

##PLotting the Temperature
plt.plot(time_MD, temperature)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('Temperature_vs_Time_big.png')

##Plotting the Energy
##new plot
plt.figure()
plt.plot(time_MD, energy)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Total Energy vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('Energy_vs_Time_big.png')

##Plotting the MSD
plt.figure()
plt.plot(time_MD, msd)
plt.xlabel('Time')
plt.ylabel('MSD')
plt.title('Mean Square Displacement vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('MSD_vs_Time_big.png')

##Plotting the VACF
plt.figure()
plt.plot(time_MD, vacf)
plt.xlabel('Time')
plt.ylabel('VACF')
plt.title('Velocity Autocorrelation Function vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('VACF_vs_Time_big.png')


##Plotting the RDF
r_in, RDF_mean, RDF_std, RDF_array=RDF_data
plt.figure()
plt.fill_between(RDF_mean, RDF_mean - RDF_std, RDF_mean + RDF_std, alpha=0.2, label='Error Range')
plt.xlabel('Distance (Bohr)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function with Error Bars')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('RDF_with_error_bars_big.png')


##Saving as txt files
np.savetxt(f"time_big_nve.txt", time_MD)
np.savetxt(f"temperature_big_nve.txt", temperature)
np.savetxt(f"energy_big_nve.txt", energy)
tot_atom=coords.shape[1]
np.savetxt(f"msd_big_nve.txt", msd)
np.savetxt(f"vacf_big_nve.txt", vacf)
with open(f"final_output_nve.txt", "w") as f:
    f.write(f"Final coordinates:\n")
    for i in range(tot_atom):
         f.write(f"Atom {i+1}: {coords[:, i]}\n")
    f.write(f"Final velocities:\n")
    for i in range(tot_atom):
        f.write(f"Atom {i+1}: {vel[:, i]}\n")

print("SCF Routine has exited")
print(qtmlogger)