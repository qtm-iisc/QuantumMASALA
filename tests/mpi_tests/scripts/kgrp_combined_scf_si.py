"""Thorough check of kgrp combined with pwgrp/bgrp: Si SCF (7 IBZ
k-points), checking BOTH total energy AND the full per-k-point eigenvalue
array (not just energy, which could hide a wrong-but-energy-neutral
mismatch) against the trusted serial reference. Also exercises UNEVEN
k-point splits (7 k-points not evenly divisible by n_kgrp), the likeliest
place for an off-by-one bug in scatter_slice-based partitioning.

Usage: mpirun -np N python3 kgrp_combined_scf_si.py N_KGRP PWGRP_SIZE
"""
import os
import sys

import numpy as np

from qtm import qtmconfig
qtmconfig.set_gpu(False)

from qtm.constants import RYDBERG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, scf
from qtm.gspace import GSpace
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.lattice import RealLattice
from qtm.config import MPI4PY_INSTALLED
from qtm.mpi import QTMComm
from qtm.mpi.gspace import DistGSpace
from qtm.pseudo import UPFv2Data

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
n_kgrp = int(sys.argv[1])
pwgrp_size = int(sys.argv[2])
dftcomm = DFTCommMod(comm_world, n_kgrp, pwgrp_size)

SYSTEM_TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "system_tests")

reallat = RealLattice.from_alat(
    alat=10.2, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]
)
si_oncv = UPFv2Data.from_file(os.path.join(SYSTEM_TESTS_DIR, "Si_ONCV_PBE-1.2.upf"))
si_atoms = BasisAtoms(
    "si", si_oncv, 28.086, reallat,
    np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]).T,
)
crystal = Crystal(reallat, [si_atoms])

mpgrid_shape = (4, 3, 1)
mpgrid_shift = (False, True, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)
assert kpts.numkpts == 7

ecut_wfn = 5 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho_serial = GSpace(crystal.recilat, ecut_rho)
grho = DistGSpace.from_dftcomm(dftcomm, grho_serial)
gwfn = grho

numbnd = crystal.numel // 2
out = scf(
    dftcomm, crystal, kpts, grho, gwfn, numbnd,
    is_spin=False, is_noncolin=False, symm_rho=True,
    occ_typ="fixed", conv_thr=1e-8 * RYDBERG, diago_thr_init=1e-2 * RYDBERG,
    iter_printer=lambda *a, **k: None,
)
scf_converged, rho, l_wfn_kgrp, en = out
assert scf_converged

REF_TOTAL_ENERGY = -7.640125014954195
REF_EIGENVALUES = np.array(
    [
        [-0.17022504, 0.16252761, 0.22588733, 0.22588734],
        [-0.15295112, 0.14666559, 0.18508893, 0.19158568],
        [-0.08615121, 0.01177918, 0.14573473, 0.17641629],
        [-0.12734933, 0.07916298, 0.13864487, 0.20814514],
        [-0.09497278, -0.00494601, 0.19909315, 0.19909315],
        [-0.07740766, 0.02630399, 0.11573031, 0.16016657],
        [-0.02246724, -0.02246723, 0.13688391, 0.13688391],
    ]
)

local_evl = np.array([wfn[0].evl for wfn in l_wfn_kgrp])
# Deduplicate by i_kgrp (every rank within the same kgrp -- possibly many,
# once bgrp/pwgrp are active -- holds an identical, fully-reconstructed
# local_evl by the time SCF returns), then sort by i_kgrp and concatenate:
# i_kgrp indexes a contiguous, increasing slice of the global k-point list
# (see scatter_slice(kpts.numkpts, n_kgrp, i_kgrp) in scf.py), so this
# reassembles the full, correctly-ordered k-point list regardless of how
# many ranks share each kgrp.
tagged = comm_world.allgather((dftcomm.i_kgrp, local_evl))
by_kgrp = {i_kgrp: arr for i_kgrp, arr in tagged}
all_evl = np.concatenate([by_kgrp[i] for i in sorted(by_kgrp)], axis=0)

if comm_world.rank == 0:
    en_err = abs(en.total - REF_TOTAL_ENERGY)
    evl_err = np.max(np.abs(all_evl - REF_EIGENVALUES))
    assert all_evl.shape == REF_EIGENVALUES.shape, (
        f"gathered eigenvalue shape {all_evl.shape} != reference "
        f"{REF_EIGENVALUES.shape}"
    )
    assert en_err < 1e-5, f"total energy diverged: err={en_err:.3e}"
    assert evl_err < 1e-4, f"eigenvalues diverged: err={evl_err:.3e}"
    print(
        f"OK: np={comm_world.size} n_kgrp={n_kgrp} pwgrp_size={pwgrp_size} "
        f"(n_bgrp={dftcomm.n_bgrp}), energy err={en_err:.3e}, "
        f"eigenvalue err={evl_err:.3e}"
    )
