import numpy as np
from qtm.config import qtmconfig, MPI4PY_INSTALLED
from qtm.constants import ELECTRONVOLT, RYDBERG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, scf
from qtm.gspace import GSpace
from qtm.io_utils.dft_printers import print_scf_status
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.lattice import RealLattice
from qtm.logger import qtmlogger
from qtm.mpi import QTMComm
from qtm.pseudo import UPFv2Data

# qtmconfig.fft_backend = 'mkl_fft'
if qtmconfig.gpu_enabled:
    qtmconfig.fft_backend = "cupy"

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

# Lattice
reallat = RealLattice.from_alat(
    alat=8, a2=[0.570726115, 0.570726115, 1.031099100], a1=[0.570726115, 1.031099100, 0.570726115], a3=[1.031099100, 0.570726115, 00.570726115]  # Bohr
)

# Atom Basis
Co_oncv = UPFv2Data.from_file("./Co_ONCV_PBE-1.2_wfc.upf")
O_oncv = UPFv2Data.from_file("./O_ONCV_PBE-1.2.upf")


U_param = {
    "nl": "3d", 
    "U": 6.3 * ELECTRONVOLT,
    "proj": "atomic", 
    "type": "d" #Dudarev
}

Co1_atoms = BasisAtoms("Co1", Co_oncv, 58.933194, reallat, np.array([[0.0, 0.0, 0.0]]).T, 0.5, [U_param])
Co2_atoms = BasisAtoms("Co2", Co_oncv, 58.933194, reallat, np.array([[0.5, 0.5, 0.5]]).T, -0.5, [U_param])
O_atoms = BasisAtoms("O", O_oncv, 15.999, reallat, np.array([[0.25, 0.25, 0.25], [0.75,0.75,0.75]]).T)

# Represents the crystal
crystal = Crystal(
    reallat,
    [
        Co1_atoms, Co2_atoms, O_atoms
    ],
)

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (3,3,3)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 50 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

# -----Spin-polarized (collinear) calculation-----
is_spin, is_noncolin = True, False
# Starting with asymmetric spin distribution else convergence may yield only
# non-magnetized states
mag_start = [0.5, -0.5, 0]
numbnd = 28 # Ensure adequate # of bands if system is not an insulator

occ = "smearing"
smear_typ = "gauss"
e_temp = 0.01 * RYDBERG

conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

print("diago_thr_init :", diago_thr_init)  # debug statement
print("e_temp :", e_temp)  # debug statement
print("conv_thr :", conv_thr)  # debug statement
print("smear_typ :", smear_typ)  # debug statement
print("is_spin :", is_spin)  # debug statement
print("is_noncolin :", is_noncolin)  # debug statement
print("ecut_wfn :", ecut_wfn)  # debug statement
print("ecut_rho :", ecut_rho)  # debug statement


out = scf(
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin,
    is_noncolin,
    mix_beta=0.1,
    mix_dim=2,
    rho_start=mag_start,
    occ_typ="smear",
    smear_typ="gauss",
    e_temp=e_temp,
    conv_thr=conv_thr,
    diago_thr_init=diago_thr_init,
    iter_printer=print_scf_status,
)

scf_converged, rho, l_wfn_kgrp, en = out

if comm_world.rank == 0:
    print("SCF Routine has exited")
    print(qtmlogger)