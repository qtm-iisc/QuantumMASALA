import os

import numpy as np

from qtm.config import qtmconfig
from qtm.constants import ELECTRONVOLT, RYDBERG
from qtm.containers.wavefun import get_WavefunG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, scf
from qtm.gspace import GSpace
from qtm.io_utils.dft_printers import print_scf_status
from qtm.kpts import KList
from qtm.lattice import RealLattice
from qtm.logger import qtmlogger
from qtm.mpi import QTMComm
from qtm.mpi.gspace import DistGSpace
from qtm.pseudo import UPFv2Data
from qtm.tddft_gamma.optical import dipole_response, dipole_spectrum

# qtmconfig.fft_backend = 'mkl_fft'

DEBUGGING = True
qtmconfig.set_gpu(False)


from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)

# Lattice
reallat = RealLattice.from_alat(
    alat=30.0, a1=[1.0, 0.0, 0.0], a2=[0.0, 1.0, 0.0], a3=[0.0, 0.0, 1.0]  # Bohr
)


# Atom Basis
c_oncv = UPFv2Data.from_file("C_ONCV_PBE-1.2.upf")
h_oncv = UPFv2Data.from_file("H_ONCV_PBE-1.2.upf")

# C atom at the center of the cell
c_atoms = BasisAtoms.from_angstrom(
    "C", c_oncv, 12.011, reallat, 0.529177 * np.array([15.0, 15.0, 15.0])
)
coords_ang = 0.642814093
h_atoms = coords_ang * np.array([[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]])
# Shift the H atoms to the center of the cell
h_atoms += 0.529177 * 15.0 * np.ones_like(h_atoms)
h_atoms = BasisAtoms.from_angstrom("H", h_oncv, 1.000, reallat, *h_atoms)


crystal = Crystal(reallat, [c_atoms, h_atoms])
kpts = KList.gamma(crystal.recilat)
print(kpts.numkpts)


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


is_spin, is_noncolin = False, False
numbnd = crystal.numel // 2
occ = "fixed"
conv_thr = 1e-10 * RYDBERG
diago_thr_init = 1e-5 * RYDBERG


out = scf(
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin,
    is_noncolin,
    occ_typ=occ,
    conv_thr=conv_thr,
    diago_thr_init=diago_thr_init,
    iter_printer=print_scf_status,
)

scf_converged, rho, l_wfn_kgrp, en = out

WavefunG = get_WavefunG(l_wfn_kgrp[0][0].gkspc, 1)

if comm_world.rank == 0:
    print("SCF Routine has exited")
    print(qtmlogger)


from os import path, remove

for fname in ["rho.npy", "wfn.npz"]:
    if path.exists(fname) and path.isfile(fname):
        remove(fname)

# %%

# -----------------------
# BEGIN TDDFT CALCULATION
# -----------------------
gamma_efield_kick = 1e-4  # Electric field kick (in z-direction) in Hartree atomic units, 0.0018709241 Ry/e_Ry/Bohr = 0.01 Ha/e_Ha/Angstrom
time_step = (
    0.1  # Time in Hartree atomic units 1 Hartree a.u. = 2.4188843265864(26)×10−17 s.
)
# Reference calculation (ce-tddft) had 2.4 attosecond time step.
numsteps = 10_002

qtmconfig.tddft_prop_method = "etrs"
qtmconfig.tddft_exp_method = "taylor"


# Pretty-print the input parameters for tddft
if comm_world.rank == 0:
    print("TDDFT Parameters:")
    print("Electric field kick:", gamma_efield_kick)
    print("Time step:", time_step)
    print("Number of steps:", numsteps)
    print("Propagation method:", qtmconfig.tddft_prop_method)
    print("Exponential evaluation method:", qtmconfig.tddft_exp_method)

dip_z = dipole_response(
    comm_world, crystal, l_wfn_kgrp, time_step, numsteps, gamma_efield_kick, "z"
)

# Transforming the response to energy spectrum

en_start = 0
en_end = 40 * ELECTRONVOLT
en_step = en_end / len(dip_z)
damp_func = "gauss"
dip_en = dipole_spectrum(
    dip_t=dip_z,
    time_step=time_step,
    damp_func=damp_func,
    # damp_fac=5e-3,
    en_end=en_end,
    en_step=en_step,
)
import matplotlib.pyplot as plt
plt.plot(dip_en[0] / ELECTRONVOLT, np.imag(dip_en[1])[:, 2])
plt.savefig("dipz.png")

fname = "dipz.npy"
if os.path.exists(fname):
    if os.path.isfile(fname):
        os.remove(fname)
np.save(fname, dip_z)
