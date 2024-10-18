import os

from qtm.config import qtmconfig
from qtm.constants import ELECTRONVOLT, RYDBERG
from qtm.containers.wavefun import get_WavefunG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, scf
from qtm.gspace import GSpace
from qtm.io_utils.dft_printers import print_scf_status
from qtm.kpts import KList, gen_monkhorst_pack_grid
from qtm.lattice import RealLattice
from qtm.logger import qtmlogger
from qtm.mpi import QTMComm
from qtm.pseudo import UPFv2Data
from qtm.tddft_gamma.optical import dipole_response

# qtmconfig.fft_backend = 'mkl_fft'


if qtmconfig.gpu_enabled:
    qtmconfig.fft_backend = "cupy"

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

# Lattice
# print("WARNING!! : Please revert the alat back to 32 Bohr")
reallat = RealLattice.from_alat(
    alat=32.0, a1=[1.0, 0.0, 0.0], a2=[0.0, 1.0, 0.0], a3=[0.0, 0.0, 0.83]  # Bohr
)


# Atom Basis
c_oncv = UPFv2Data.from_file("C_ONCV_PBE-1.2.upf")
h_oncv = UPFv2Data.from_file("H_ONCV_PBE-1.2.upf")

c_atoms = BasisAtoms.from_angstrom(
    "C",
    c_oncv,
    12.011,
    reallat,
    (5.633200899, 6.320861303, 5.000000000),
    (6.847051545, 8.422621957, 5.000000000),
    (8.060751351, 7.721904557, 5.000000000),
    (8.060707879, 6.320636665, 5.000000000),
    (6.846898786, 5.620067381, 5.000000000),
    (5.633279551, 7.722134449, 5.000000000),
)

h_atoms = BasisAtoms.from_angstrom(
    "H",
    h_oncv,
    1.008,
    reallat,
    (6.847254360, 9.512254789, 5.000000000),
    (9.004364510, 8.266639340, 5.000000000),
    (9.004297495, 5.775895755, 5.000000000),
    (6.846845929, 4.530522778, 5.000000000),
    (4.689556006, 5.776237709, 5.000000000),
    (4.689791688, 8.267023318, 5.000000000),
)

crystal = Crystal(reallat, [c_atoms, h_atoms])
kpts = KList.gamma(crystal.recilat)
print(kpts.numkpts)


# -----Setting up G-Space of calculation-----
ecut_wfn = 40 * RYDBERG
ecut_rho = 4 * ecut_wfn
gspc_rho = GSpace(crystal.recilat, ecut_rho)
gspc_wfn = gspc_rho


is_spin, is_noncolin = False, False
numbnd = crystal.numel // 2
occ = "fixed"
conv_thr = 1e-10 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

# occ = 'smear'
# smear_typ = 'gauss'
# e_temp = 1E-2 * RYDBERG


print("diago_thr_init :", diago_thr_init)  # debug statement
# print('e_temp :', e_temp) #debug statement
print("conv_thr :", conv_thr)  # debug statement
# print('smear_typ :', smear_typ) #debug statement
print("is_spin :", is_spin)  # debug statement
print("is_noncolin :", is_noncolin)  # debug statement
print("ecut_wfn :", ecut_wfn)  # debug statement
print("ecut_rho :", ecut_rho)  # debug statement


out = scf(
    dftcomm,
    crystal,
    kpts,
    gspc_rho,
    gspc_wfn,
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

    wfn_gamma = l_wfn_kgrp[0]
    from os import path, remove

    import numpy as np

    for fname in ["rho.npy", "wfn.npz"]:
        if path.exists(fname) and path.isfile(fname):
            remove(fname)

    print(len(wfn_gamma))
    # print(wfn_gamma[0].__doc__())
    # print("Saving charge density and wavefun data to 'rho.npy' and 'wfn.npz'")
    # np.savez('wfn.npz', evc_gk=wfn_gamma[0].evc_gk, evl=wfn_gamma[0].evl,
    #          occ=wfn_gamma[0].occ)
    # np.save('rho.npy', rho.g)


# -----------------------
# BEGIN TDDFT CALCULATION
# -----------------------

gamma = 1e-4
time_step = 0.05
numsteps = 10

dip_z = dipole_response(
    comm_world, crystal, rho, l_wfn_kgrp, time_step, numsteps, gamma, "z"
)

fname = "dipz.npy"
if os.path.exists(fname) and os.path.isfile(fname):
    os.remove(fname)
np.save(fname, dip_z)
