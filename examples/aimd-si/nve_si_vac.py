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
from qtm.MD import NVE_MD

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

# Only G-space parallelization
# K-point and/or band parallelization along with G-space parallelization is currently broken.
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)

parser = argparse.ArgumentParser()
parser.add_argument("supercell_size", help="Side length of the supercell", type=int)

args = parser.parse_args()
supercell_size = args.supercell_size

alat=10.2*supercell_size
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
crystal_supercell=crystal_unit.gen_supercell([supercell_size] * 3)
##print the crystal coordinates of the supercell
#print("The crystal coordinates of the supercell", crystal_supercell.l_atoms[0].r_alat)
r_alat_supercell=crystal_supercell.l_atoms[0].r_alat.T

#print("the original coordinates are", r_alat_supercell)
##relaxed coordinates
data=[[-0.24388464, -0.07177099, -0.48701147, -0.25324947, -0.68270201,
        -0.30767731, -0.1186068 , -0.44449295, -0.17766944, -0.56923016,
        -0.46250233, -0.41610712, -0.0564199 , -0.44546106, -0.20260633,
        -0.65612672, -0.44080038, -0.42019349, -0.28382821, -0.62999969,
        -0.24978634, -0.61281531, -0.67588949, -0.51872129, -0.16363928,
        -0.61821331, -0.37838157, -0.88922691, -0.54497082, -0.36700514,
        -0.26352025, -0.56771428, -0.40580758, -0.77039716, -0.4872718 ,
        -0.62586747, -0.35742866, -0.74747766, -0.52906054, -0.80227005,
        -0.74190857, -0.63759434, -0.32149038, -0.69538796, -0.46968872,
        -1.04385878, -0.73135056, -0.60977657, -0.3616685 , -0.81820978,
        -0.49485122, -0.99824172, -0.81512472],
       [ 0.36112038, -0.02771264,  0.35710896,  0.10458031,  0.63369769,
         0.31174533,  0.1528884 ,  0.66097741,  0.40133309,  0.73433568,
         0.60237009,  0.57729272,  0.34968672,  0.81041283,  0.48303126,
         1.00757982,  0.82002823,  0.23791708,  0.08039788,  0.49911655,
         0.21770605,  0.6437533 ,  0.37127906,  0.40384953,  0.241447  ,
         0.49469792,  0.30701878,  0.65128549,  0.48773858,  0.68503063,
         0.44586196,  0.7342162 ,  0.42842331,  0.86750294,  0.66617742,
         0.30541355,  0.06642869,  0.44458914,  0.16589775,  0.57862094,
         0.48501766,  0.53575297,  0.13143859,  0.67058162,  0.24710134,
         0.71476121,  0.71126898,  0.61124263,  0.4606158 ,  0.73242937,
         0.51669173,  1.0275412 ,  0.7408899 ],
       [ 0.33476655, -0.02997953,  0.26625432, -0.02193522,  0.42711638,
         0.07372501,  0.14848329,  0.43789407,  0.21353501,  0.45932047,
         0.30592988,  0.55406612,  0.54775472,  0.63256524,  0.40757301,
         0.69567591,  0.23620319,  0.46790408,  0.12279701,  0.40612738,
         0.17928359,  0.56882003,  0.13142502,  0.61766005,  0.33689689,
         0.54845004,  0.34561075,  0.69122397,  0.28925067,  0.76060393,
         0.54277684,  0.7804941 ,  0.52906506,  0.74330385,  0.66808527,
         0.6028384 ,  0.34576236,  0.59709732,  0.4005999 ,  0.60691539,
         0.31623621,  0.77819684,  0.47283745,  0.74761039,  0.61056186,
         0.76147497,  0.61461539,  1.00939428,  0.67083449,  0.94361335,
         0.70835537,  0.85577595,  0.79729863]]
data=np.array(data).T

N=data

#print("the new coordinates in alat units", N)

si_atoms_supercell = BasisAtoms.from_alat(
    "si",
    si_oncv,
    28.086,
    reallat,
    N,  # Fractional coordinates
)

crystal = Crystal(reallat, [si_atoms_supercell]) 

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

#print("the type of grho is", type(grho))    

numbnd = int(2.5*(crystal.numel // 2)) # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

#How many atoms in the cell?
tot_num = np.sum([sp.numatoms for sp in crystal.l_atoms])
extra=int(max(4, 2*tot_num))
numbnd = int(2*tot_num)  +extra # Ensure adequate # of bands if system is not an insulator
conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-4 * RYDBERG


##Smearing
occ_typ="smear"
smear_typ = 'gauss'
e_temp = 1E-2 * RYDBERG

steps=5000
dt=40
max_t=steps*dt
T_init=5

from time import perf_counter

initial_time=perf_counter()

out = NVE_MD(dftcomm, crystal, max_t, dt, T_init, mpgrid_shape, mpgrid_shift , ecut_wfn,
          numbnd, is_spin=False, is_noncolin=False,
          symm_rho=False, rho_start=None, occ_typ='smear',
            smear_typ='gauss', e_temp=1E-2 * RYDBERG,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

coords, time, temperature, energy= out


final_time=perf_counter()
print("The time taken is", final_time-initial_time)
print("Approx time per iteration", (final_time-initial_time)/steps)

##PLotting the Temperature
plt.plot(time, temperature)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('Temperature_vs_Time_big_vac.png')

##Plotting the Energy
##new plot
plt.figure()
plt.plot(time, energy)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Total Energy vs Time')
plt.legend(f"each time step represents {dt} atomic units")
plt.savefig('Energy_vs_Time_big_vac.png')


##Saving as txt files
np.savetxt("time_big.txt", time)
np.savetxt("temperature_big.txt", temperature)
np.savetxt("energy_big.txt", energy)



print("SCF Routine has exited")
print(qtmlogger)
