"""Driver (not collected by pytest): run under `mpirun -np N python3
bgrp_tddft_short.py PWGRP_SIZE` by `tests/mpi_tests/test_dist_bgrp.py`.

Checks that a short CH4 TaylorExp dipole-response run reproduces the
serial reference once SCF has run with band groups active (`n_bgrp =
(N // PWGRP_SIZE) > 1`, via `DFTCommMod(comm_world, 1, PWGRP_SIZE)`) --
band groups only affect Davidson's internal band-parallel work (see
`bgrp_scf_ch4.py`); by the time SCF returns, every rank holds the full,
reconstituted band set, so TDDFT propagation from it should reproduce the
usual pwgrp-only behavior even when redundantly repeated across multiple
band groups. See `pwgrp_tddft_short.py` for why this is asserted with a
loose (1e-3) tolerance rather than machine precision (SCF's Davidson
eigensolver is not bit-reproducible across different MPI layouts, and the
resulting tiny differences get amplified somewhat by 5 steps of nonlinear
real-time dynamics -- this is expected numerical sensitivity, not a bug).
"""
import os
import sys

import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import RYDBERG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, KSWfn, scf
from qtm.gspace import GSpace
from qtm.kpts import KList
from qtm.lattice import RealLattice
from qtm.config import MPI4PY_INSTALLED
from qtm.mpi import QTMComm
from qtm.mpi.gspace import DistGSpace
from qtm.pseudo import UPFv2Data
from qtm.tddft_gamma.optical import dipole_response

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
pwgrp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
dftcomm = DFTCommMod(comm_world, 1, pwgrp_size)

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
    conv_thr=1e-10 * RYDBERG, diago_thr_init=1e-7 * RYDBERG,
    iter_printer=lambda *a, **k: None,
)
scf_converged, rho, l_wfn_kgrp, en = out
assert scf_converged

gamma_efield_kick = 1e-4
dt = 0.1
numsteps = 5
qtmconfig.tddft_prop_method = "etrs"
qtmconfig.tddft_exp_method = "taylor"

# Reference from tests/mpi_tests/scripts/pwgrp_tddft_short.py's serial
# (np=1) computation of this exact same setup.
REF_FINAL_DIPZ = -5.050014356462959

dip_z = np.array(
    dipole_response(
        comm_world, crystal, l_wfn_kgrp, dt, numsteps, gamma_efield_kick, "z",
        write_freq=-1,
    )
)

if comm_world.rank == 0:
    final = dip_z[-1, 2].real
    err = abs(final - REF_FINAL_DIPZ)
    assert err < 1e-3, (
        f"bgrp TDDFT (np={comm_world.size}, pwgrp_size={pwgrp_size}, "
        f"n_bgrp={dftcomm.n_bgrp}) diverged from serial reference: "
        f"got {final!r}, expected {REF_FINAL_DIPZ!r} (err={err:.3e})"
    )
    print(
        f"OK: np={comm_world.size} pwgrp_size={pwgrp_size} "
        f"(n_bgrp={dftcomm.n_bgrp}), taylor err={err:.3e}"
    )
