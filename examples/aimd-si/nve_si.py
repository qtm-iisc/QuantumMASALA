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
from qtm.force import force, force_ewald, force_local, force_nonloc
from qtm.MD import NVE_MD, RDist

from qtm.io_utils.dft_printers import print_scf_status

import argparse

from qtm.config import qtmconfig
from qtm.logger import qtmlogger

initial_time=time.time()

# qtmconfig.fft_backend = "pyfftw"
qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
#print("before starting nve calculation", "the size of the communicator is", comm_world.size)
# Only G-space parallelization
# K-point and/or band parallelization along with G-space parallelization is currently broken.
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)
#print("before starting nve calculation", "the size of the communicator is", dftcomm.image_comm.size)

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
##print the crystal coordinates of the supercell
#print("The crystal coordinates of the supercell", crystal_supercell.l_atoms[0].r_alat)
r_alat_supercell=crystal_supercell.l_atoms[0].r_alat.T
r_cart_supercell=crystal_supercell.l_atoms[0].r_cart.T


if dftcomm.image_comm.rank==0: 
    print("the original coordinates are\n", r_alat_supercell, "\n")
    print("the original coordinates in cartesian are\n", r_cart_supercell, "\n")

##relaxed coordinates
'''data=[[-0.25360773, -0.09576083, -0.50259631, -0.22927595, -0.67175465,
        -0.31229879, -0.32129343, -0.05276206, -0.3994334 , -0.19186968,
        -0.53585398, -0.38089773, -0.37858821, -0.05165412, -0.4343018 ,
        -0.23007814, -0.64720126, -0.46900115, -0.39635321, -0.27483042,
        -0.61399092, -0.25142682, -0.6916394 , -0.63902247, -0.44810428,
        -0.26198595, -0.55302484, -0.40305318, -0.91665914, -0.51440941,
        -0.40270455, -0.19509245, -0.55250662, -0.3297184 , -0.84794926,
        -0.54023952, -0.60136805, -0.36770041, -0.65370476, -0.50952152,
        -0.78156914, -0.71964801, -0.67341941, -0.31284516, -0.74509969,
        -0.46528159, -1.00167946, -0.80738339, -0.58115541, -0.41521609,
        -0.86515377, -0.62632375, -0.99706668, -0.84812515],
       [ 0.36298086, -0.0463367 ,  0.41451426,  0.12028743,  0.6553405 ,
         0.35574497,  0.4603902 ,  0.16474612,  0.6559331 ,  0.32214156,
         0.71741983,  0.58282701,  0.60678768,  0.32491196,  0.84217852,
         0.5953979 ,  0.90067268,  0.89092118,  0.24000518,  0.05214375,
         0.52587594,  0.20488464,  0.6588982 ,  0.27377287,  0.41432509,
         0.23639394,  0.49182704,  0.33587029,  0.69221383,  0.54842262,
         0.71807278,  0.47484754,  0.67968431,  0.47561421,  0.89782409,
         0.62564993,  0.20921706,  0.11987816,  0.46533146,  0.17182034,
         0.49464402,  0.45792772,  0.46063849,  0.11176017,  0.6413124 ,
         0.27521787,  0.75951295,  0.59887377,  0.60249222,  0.50182068,
         0.79468602,  0.54509673,  1.02674633,  0.75231692],
       [ 0.27916227, -0.01945097,  0.23112428,  0.08094686,  0.43346464,
         0.04813686,  0.39055845,  0.20628016,  0.45653402,  0.12993391,
         0.44821185,  0.33399546,  0.59154754,  0.54188989,  0.62251872,
         0.42307242,  0.70785126,  0.37156972,  0.4298655 ,  0.21942665,
         0.39005534,  0.20527984,  0.57805686,  0.14735964,  0.58128313,
         0.35109969,  0.51888339,  0.31230403,  0.65898593,  0.28337513,
         0.70293375,  0.5880235 ,  0.71735031,  0.53310487,  0.78595346,
         0.57681239,  0.57340545,  0.32014835,  0.61836223,  0.35914875,
         0.5537072 ,  0.31788079,  0.85537208,  0.46725145,  0.77909767,
         0.55353992,  0.81000004,  0.64698277,  0.96561542,  0.69427326,
         0.92543345,  0.74384706,  0.81771324,  0.78211051]]
data=np.array(data).T'''

#print("the new coordinates in alat units", N)

'''si_atoms_supercell = BasisAtoms.from_alat(
    "si",
    si_oncv,
    28.086,
    reallat,
    N,  # Fractional coordinates
)'''

#crystal = Crystal(reallat, [crystal_supercell]) 
crystal=crystal_supercell

 # Represents the crystal

#crystal = crystal.gen_supercell([supercell_size] * 3)
##We want to print the coordinates of the Si atms
#print("Si basis", si_basis.r_alat)
#coordinates=generate_coordinates(supercell_size).T
## Set this as the new coordinates of the basis
#si_basis.r_cart=coordinates
#print("new coordinates", si_basis.r_cart)
# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (1, 1, 1)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 10 * RYDBERG
ecut_rho = 4 *ecut_wfn
grho_serial = GSpace(crystal.recilat, ecut_rho)

# If G-space parallelization is not required, use the serial G-space object
#print("N_pwgrp", dftcomm.n_pwgrp)
#print("Image_comm_size", dftcomm.image_comm.size)
if dftcomm.n_pwgrp == dftcomm.image_comm.size:  
    grho = grho_serial
else:
    grho = DistGSpace(comm_world, grho_serial)
gwfn = grho

print("the type of grho is", type(grho))    

numbnd = int(1.2*(crystal.numel // 2)) # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

#How many atoms in the cell?
#numbnd = int(2*tot_num)  +extra # Ensure adequate # of bands if system is not an insulator
conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-4 * RYDBERG

##Smearing

steps=1000
dt=20
mixing_beta=0.3
max_t=steps*dt
T_init=300
smear_typ='gauss'
e_temp=1E-2 * RYDBERG
occ_typ='smear'

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
    print(flush=True)

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

out = NVE_MD(dftcomm, crystal, max_t, dt, T_init, kpts, grho, gwfn, ecut_wfn,
          numbnd, is_spin=False, is_noncolin=False,
          symm_rho=False, rho_start=None, occ_typ='smear',
            smear_typ='gauss', e_temp=1E-2 * RYDBERG,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

coords, time, temperature, energy, msd, vel= out

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
plt.plot(time, temperature)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('Temperature_vs_Time_big.png')

##Plotting the Energy
##new plot
plt.figure()
plt.plot(time, energy)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Total Energy vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('Energy_vs_Time_big.png')

##Plotting the MSD
plt.figure()
plt.plot(time, msd)
plt.xlabel('Time')
plt.ylabel('MSD')
plt.title('Mean Square Displacement vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('MSD_vs_Time_big.png')


##Saving as txt files
np.savetxt("time_big.txt", time)
np.savetxt("temperature_big.txt", temperature)
np.savetxt("energy_big.txt", energy)
tot_atom=coords.shape[1]
np.savetxt("msd_big.txt", msd)
with open("final_output.txt", "w") as f:
    f.write("Final coordinates:\n")
    for i in range(tot_atom):
         f.write(f"Atom {i+1}: {coords[:, i]}\n")
    f.write("Final velocities:\n")
    for i in range(tot_atom):
        f.write(f"Atom {i+1}: {vel[:, i]}\n")

print("SCF Routine has exited")
print(qtmlogger)
