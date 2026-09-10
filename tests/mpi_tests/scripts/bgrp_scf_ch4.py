"""Driver (not collected by pytest): run under `mpirun -np N python3
bgrp_scf_ch4.py PWGRP_SIZE` by `tests/mpi_tests/test_dist_bgrp.py`.

Checks band-group (bgrp) MPI parallelism for SCF -- a third
parallelization axis, alongside pwgrp and kgrp, that `DFTCommMod` derives
as `n_bgrp = n_pwgrp // n_kgrp` whenever `n_kgrp * pwgrp_size <
image_comm.size`. `N // PWGRP_SIZE` band groups each handle a subset of
`numbnd` bands (see `qtm.dft.eigsolve.davidson`'s use of
`dftcomm.pwgrp_inter_kgrp`), reconstituting the full band set via
Allgatherv before SCF returns.

Builds the distributed G-space via `DistGSpace.from_dftcomm`, which
resolves to `dftcomm.pwgrp_intra` -- the only communicator that is always
EXACTLY one plane-wave group. `comm_world` spans MULTIPLE separate
plane-wave groups (one per band group) once `n_bgrp>1`; distributing
G-vectors across it directly crashes with `MPI_ERR_TRUNCATE`.
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
from qtm.kpts import KList
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
pwgrp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
dftcomm = DFTCommMod(comm_world, 1, pwgrp_size)  # n_kgrp=1 -> the rest is bgrp/pwgrp

SYSTEM_TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "system_tests")

reallat = RealLattice.from_alat(
    alat=30.0, a1=[1.0, 0.0, 0.0], a2=[0.0, 1.0, 0.0], a3=[0.0, 0.0, 1.0]
)
c_oncv = UPFv2Data.from_file(os.path.join(SYSTEM_TESTS_DIR, "C_ONCV_PBE-1.2.upf"))
h_oncv = UPFv2Data.from_file(os.path.join(SYSTEM_TESTS_DIR, "H_ONCV_PBE-1.2.upf"))

c_atoms = BasisAtoms.from_angstrom(
    "C", c_oncv, 12.011, reallat, 0.529177 * np.array([15.0, 15.0, 15.0])
)
coords_ang = 0.642814093
h_atoms = coords_ang * np.array([[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]])
h_atoms += 0.529177 * 15.0 * np.ones_like(h_atoms)
h_atoms = BasisAtoms.from_angstrom("H", h_oncv, 1.000, reallat, *h_atoms)

crystal = Crystal(reallat, [c_atoms, h_atoms])
kpts = KList.gamma(crystal.recilat)

ecut_wfn = 5 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho_serial = GSpace(crystal.recilat, ecut_rho)
grho = DistGSpace.from_dftcomm(dftcomm, grho_serial)
gwfn = grho

numbnd = crystal.numel // 2
out = scf(
    dftcomm, crystal, kpts, grho, gwfn, numbnd,
    is_spin=False, is_noncolin=False, occ_typ="fixed",
    conv_thr=1e-12 * RYDBERG, diago_thr_init=1e-9 * RYDBERG,
    iter_printer=lambda *a, **k: None,
)
scf_converged, rho, l_wfn_kgrp, en = out
assert scf_converged

# Reference from the trusted serial (single-process) CH4 SCF; see
# tests/mpi_tests/scripts/pwgrp_tddft_short.py for the same value used as
# the basis of a TDDFT reference too.
REF_TOTAL_ENERGY = -7.049039627

if comm_world.rank == 0:
    err = abs(en.total - REF_TOTAL_ENERGY)
    assert err < 1e-6, f"bgrp SCF total energy diverged: err={err:.3e}"
    print(
        f"OK: np={comm_world.size} pwgrp_size={pwgrp_size} "
        f"(n_bgrp={dftcomm.n_bgrp}), energy err={err:.3e}"
    )
