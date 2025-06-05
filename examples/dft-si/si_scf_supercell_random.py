
import numpy as np
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

# Only G-space parallelization
# K-point and/or band parallelization along with G-space parallelization is currently broken.
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)

parser = argparse.ArgumentParser()
parser.add_argument("supercell_size", help="Side length of the supercell", type=int)

args = parser.parse_args()
supercell_size = args.supercell_size

alat=10.2
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

crystal = Crystal(reallat, [si_atoms]) 
crystal_supercell=crystal.gen_supercell([supercell_size] * 3)
reallat_supercell=crystal_supercell.reallat
##print the crystal coordinates of the supercell
#print("The crystal coordinates of the supercell", crystal_supercell.l_atoms[0].r_alat)
r_alat_supercell=crystal_supercell.l_atoms[0].r_alat.T
numatom=r_alat_supercell.shape[0]

#print("the original coordinates are", r_alat_supercell)

data=[[ 6.58943879e-04,6.91497266e-04,7.48515032e-05],
 [-6.05728097e-05, -8.75943473e-04, -7.04679950e-04],
 [-6.10954497e-04, -7.56690520e-04, -8.03117566e-04],
 [-9.45313375e-04, -9.18127002e-04,4.66155557e-04],
 [-5.54215340e-04, -3.11150267e-04,6.97699613e-04],
 [ 4.70218205e-04, -2.94599103e-04,5.82396611e-04],
 [ 1.34223693e-04,5.39662464e-04, -7.56884917e-04],
 [ 1.46562007e-04, -4.75724968e-04,3.49339605e-05],
 [ 3.69045709e-05,5.57544205e-04, -2.28853210e-04],
 [-9.00268630e-05, -5.47027040e-04,1.76136806e-04],
 [ 4.61637614e-04, -9.79828799e-04, -1.53985018e-04],
 [-2.81704763e-04,9.31582408e-04,8.38253258e-04],
 [-2.83497159e-04, -4.75322115e-04, -2.09080901e-04],
 [ 6.31048459e-04, -4.72329903e-04,8.38034461e-04],
 [ 4.90094430e-04,5.37888451e-04,1.71644742e-04],
 [ 1.10833347e-04,8.81252041e-05,5.23560885e-04],
 [-5.04979146e-04,4.49471438e-04,5.75469663e-04],
 [-9.10478528e-05,8.85532305e-04,1.91818065e-04],
 [ 4.73158124e-04, -9.14054407e-04, -6.20230266e-04],
 [-7.51088497e-04,6.05364535e-04, -2.27549946e-04],
 [ 1.47776811e-05,9.24451428e-04, -7.83801622e-04],
 [ 6.93567368e-04, -2.98025129e-04, -3.18008522e-05],
 [ 6.61267454e-04,2.22968662e-04,8.32347226e-04],
 [-8.07183488e-04, -1.18307189e-04, -9.11752326e-04],
 [-3.28766544e-04, -9.00191478e-04, -5.56297740e-04],
 [-6.12949076e-04,4.61322052e-05, -6.24005589e-04],
 [ 5.57522831e-04, -6.27987365e-04, -8.37730046e-04],
 [-2.85304595e-04, -7.31672139e-04, -7.44141875e-04],
 [-9.74106801e-04, -5.89766002e-04,6.74298368e-04],
 [ 8.13531704e-05,1.91106733e-04, -8.15295801e-04],
 [ 8.46892006e-04,9.03127092e-04, -2.45955077e-04],
 [-6.59793927e-04,6.62838914e-04,4.50476010e-04],
 [ 7.42760612e-04, -8.01320341e-04, -3.15969379e-04],
 [ 8.52419875e-04, -9.60273012e-04,1.13681948e-04],
 [-7.97482839e-04, -3.91599552e-04, -3.26885348e-04],
 [ 3.98426986e-04,6.32619791e-04,5.24122998e-04],
 [ 3.78839087e-04, -4.99125943e-04,6.51472900e-05],
 [ 1.25583492e-04,6.68896747e-04, -8.03354387e-05],
 [ 3.94636444e-04, -1.14553966e-04, -1.74867050e-04],
 [-5.76888111e-04, -6.50713363e-04, -3.05995101e-04],
 [ 9.81393712e-04, -8.63231993e-04, -9.23577518e-04],
 [-4.37174688e-04,9.40450698e-04, -5.56705373e-04],
 [-3.84127055e-04,2.60384611e-05,1.92124696e-04],
 [ 4.15749460e-04, -4.67585213e-04, -8.85328530e-04],
 [ 9.42300802e-04, -2.92518048e-04, -2.63869872e-04],
 [ 7.44418850e-04, -9.10014530e-04,9.49943459e-05],
 [-6.26709933e-04, -3.45556276e-04,4.59098437e-04],
 [-5.09495385e-04,6.92071890e-04,6.30685829e-04],
 [ 6.63648691e-04, -1.68377664e-04,7.66466740e-04],
 [ 5.65275529e-05,9.75354510e-04, -7.27349789e-04],
 [-3.82731670e-04, -6.25633060e-04, -1.78607492e-04],
 [-4.04577925e-04, -6.72948025e-05,4.37152063e-04],
 [ 1.58613273e-04,1.07264431e-04, -9.99014418e-04],
 [-9.95646352e-04, -2.10658918e-04,9.96020775e-04]]

data=np.array(data)

N=r_alat_supercell+ 1*data

#print("the new coordinates in alat units", N)

si_atoms_supercell = BasisAtoms.from_alat(
    "si",
    si_oncv,
    28.086,
    reallat,
    N,
)

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


# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
ecut_rho = 4 *ecut_wfn


# If G-space parallelization is not required, use the serial G-space object
#print("N_pwgrp", dftcomm.n_pwgrp)
#print("Image_comm_size", dftcomm.image_comm.size)

print(flush=True)

 # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

r_alat_supercell=crystal_supercell.l_atoms[0].r_alat.T

#print("the original coordinates are", r_alat_supercell)

high=1e-3
low=-1e-3
num_itr=10
##Make a dictionary to store the arrays:

def convert_to_qe_input(coordinates):
    qe_input = ""
    for coord in coordinates:
        ##truncate the coordinates to 8 decimal places
        coord = [round(c, 9) for c in coord]
        qe_input += f"Si {coord[0]} {coord[1]} {coord[2]}\n"
    return qe_input

def generate_qe_input_file(coordinates, config_index):
    qe_input = convert_to_qe_input(coordinates)
    input_template = f"""
&control
    calculation = 'scf',
    restart_mode='from_scratch',
    prefix = 'si_{config_index}',
    tstress = .true.,
    tprnfor = .true.,
    verbosity = 'high',
    pseudo_dir = '/home/sanmitc/qe-7.3/pseudo/',
    outdir='/home/sanmitc/qe-7.3/tempdir/'
/
&system
    ibrav = 2,
    celldm(1) = 30.6,
    nat = {len(coordinates)},
    nbnd=130,
    ntyp = 1,
    ecutwfc = 25.0
    nosym=.true
    occupations= 'smearing',
    smearing= 'gaussian',
    degauss= 0.01
/
&electrons
    diagonalization='davidson'
    mixing_mode = 'plain'
    conv_thr = 1.0d-8,
    mixing_beta = 0.3,
/
ATOMIC_SPECIES
Si  28.086  Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS alat
{qe_input}
K_POINTS automatic
1 1 1 0 0 0
"""
    with open(f"si_{config_index}.in", "w") as f:
        f.write(input_template)
    return f"si_{config_index}.in"

for itr in range(num_itr):
    if dftcomm.image_comm.rank==0: 
        print("***********************************")
        print("***********************************")
        print("Iteration number", itr)
        print("***********************************")
        print("***********************************")
        print(flush=True)
        np.random.seed(None)
        data=np.random.uniform(low, high, size=r_alat_supercell.shape)

    data=comm_world.bcast(data, root=0)
    N=r_alat_supercell+ 1*data
    qe_input = convert_to_qe_input(N)
    ##Make the .in file
    qe_input_file = generate_qe_input_file(N, itr)

    data_comma= np.array2string(data, separator=',')

    if dftcomm.image_comm.rank==0: 
        print("the random data added to the crystal is", data_comma, "for iteration", itr)
        print("the new coordinates in alat units", N, "for iteration", itr)

    si_atoms_supercell = BasisAtoms.from_alat(
        "si",
        si_oncv,
        28.086,
        reallat_supercell,
        N,
    )

    crystal = Crystal(reallat_supercell, [si_atoms_supercell]) 
    numbnd = int(np.round(1.2*(crystal.numel // 2)))
    kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

    grho_serial = GSpace(crystal.recilat, ecut_rho)

    if dftcomm.n_pwgrp == dftcomm.image_comm.size:  
        grho = grho_serial
    else:
        grho = DistGSpace(comm_world, grho_serial)
    gwfn = grho

    print("the type of grho is", type(grho))    
    print(flush=True)

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
        occ_typ="smear",
        smear_typ="gauss",
        e_temp=0.01 * RYDBERG,
        conv_thr=conv_thr,
        diago_thr_init=diago_thr_init,
        iter_printer=print_scf_status,
        force_stress=True
    )
    
    scf_converged, rho, l_wfn_kgrp, en, v_loc, nloc, xc_compute= out

    if dftcomm.image_comm.rank==0:
        print("the eigen values are", l_wfn_kgrp[0][0].evl)
        print("the occupation numbers are", l_wfn_kgrp[0][0].occ)

    ##print the eigen values
    #if dftcomm.image_comm.rank==0:
        #eig=l_wfn_kgrp
    #print("SCF converged", scf_converged)
    #print(flush=True)

    initial_time=time.time()
    force_ewa=force_ewald(dftcomm=dftcomm,
                        crystal=crystal,
                        gspc=gwfn, 
                        gamma_only=False)
    

    if dftcomm.image_comm.rank==0:
        print("force ewald", force_ewa)
        print("Time taken for ewald force: ", time.time() - initial_time)
    print(flush=True)

    ##Calculation time of Local Forces
    start_time = time.time()
    force_loc=force_local(dftcomm=dftcomm,
                        cryst=crystal, 
                        gspc=gwfn, rho=rho, 
                        vloc=v_loc,
                        gamma_only=False)

    if dftcomm.image_comm.rank==0:
        print("force local", force_loc)
        print("Time taken for local force: ", time.time() - initial_time)
    print(flush=True)

    ##Calculation time of Non Local Forces
    start_time = time.time()
    force_nloc=force_nonloc(dftcomm=dftcomm,
                            numbnd=numbnd,
                            wavefun=l_wfn_kgrp, 
                            crystal=crystal,
                            nloc_dij_vkb=nloc)

    if dftcomm.image_comm.rank==0:
        print("force non local", force_nloc)
        print("Time taken for non local force: ", time.time() - initial_time)
    print(flush=True)

    #force_time=time.time()
    start_time = time.time()
    force_total, force_norm=force(dftcomm=dftcomm,
                                numbnd=numbnd,
                                wavefun=l_wfn_kgrp,
                                crystal=crystal,
                                gspc=gwfn, 
                                rho=rho,
                                vloc=v_loc,
                                nloc_dij_vkb=nloc,
                                gamma_only=False,
                                verbosity=True)

    if dftcomm.image_comm.rank==0:
        print("force total", force_total)
        print("force norm", force_norm)
        print("Time taken for force: ", time.time() - start_time)

    if comm_world.rank == 0:
        print("SCF Routine has exited")
        print(qtmlogger)

    final_time=time.time() 

    if dftcomm.image_comm.rank==0:
        print("Total time taken for the calculation", final_time-initial_time)
        print("End of iteration", itr)
        print("======================================")
        print("======================================")
        print("======================================")
        print(flush=True)


