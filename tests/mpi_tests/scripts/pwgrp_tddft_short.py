"""Driver (not collected by pytest): run under `mpirun -np N python3
pwgrp_tddft_short.py` by `tests/mpi_tests/test_dist_pwgrp.py`.

Regression guard for a BiCGSTAB-under-MPI deadlock (see
`qtm.linalg.bicgstab`): `SplitOper` (Cayley
`vloc_method`, the default) and `CrankNicolson` both solve a per-band
Cayley transform iteratively, and a naive (`scipy.sparse.linalg.bicgstab`)
solver checks convergence using each MPI rank's LOCAL residual only, which
desynchronizes the collective `matvec` calls across ranks and deadlocks
under `pwgrp_size>1` (confirmed directly: a 10-step run that takes ~2s
serially spun at 100% CPU on every rank for 90+ seconds under
`mpirun -np 2` without completing). This script runs a short (5-step) CH4
dipole response with all five propagators at whatever rank count it's
invoked with; the test wrapper bounds it with a timeout, so a
reintroduced deadlock fails the test instead of hanging CI.

`KTEExp`/`LanczosExp` have their own MPI-sensitive code path worth
covering here too: both estimate spectral bounds via
`qtm.linalg.lanczos.lanczos_bounds`, which draws its cold-start probe
vector independently per rank (`KTEExp._random_probe`) and reduces every
inner product with `pwgrp_comm.allreduce` (`_global_vdot`) -- a plausible
place for a `pwgrp_size>1`-only bug to hide the same way the BiCGSTAB one
did. The probe being random (not seeded) means the exact Chebyshev/Krylov
order taken can differ run to run and rank-count to rank-count, but the
final propagated state should still agree with the serial reference to
within the same tolerance as the other three methods regardless -- that
invariance is exactly what this test checks.
"""
import os

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
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)

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
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin=False,
    is_noncolin=False,
    occ_typ="fixed",
    conv_thr=1e-10 * RYDBERG,
    diago_thr_init=1e-7 * RYDBERG,
    iter_printer=lambda *a, **k: None,
)
scf_converged, rho, l_wfn_kgrp, en = out
assert scf_converged

gamma_efield_kick = 1e-4
dt = 0.1
numsteps = 5

# Reference final z-dipoles computed via this same script's serial (np=1)
# path. kte/lanczos draw an unseeded random probe for spectral-bound
# estimation, so their own serial value already varies run to run by
# ~1e-5 (checked directly) well inside the 1e-4 tolerance below -- this is
# expected noise in the estimate, not a correctness issue, since the
# padded bounds only affect how many Chebyshev/Krylov terms are used, not
# the accuracy of the resulting exponential approximation.
REF_FINAL_DIPZ = {
    "taylor": -5.050014356462959,
    "splitoper": -5.04793497884917,
    "cranknicolson": -5.0483522511364285,
    "kte": -5.050018484365433,
    "lanczos": -5.0500172307045395,
}


def clone_wfn(wfn):
    new = KSWfn(wfn.gkspc, wfn.k_weight, wfn.numbnd, wfn.is_noncolin)
    new.evc_gk[:] = wfn.evc_gk[:]
    new.evl[:] = wfn.evl[:]
    new.occ[:] = wfn.occ[:]
    return new


for exp_method in ["taylor", "splitoper", "cranknicolson", "kte", "lanczos"]:
    qtmconfig.tddft_prop_method = "etrs"
    qtmconfig.tddft_exp_method = exp_method
    wfn_copy = clone_wfn(l_wfn_kgrp[0][0])
    dip_z = np.array(
        dipole_response(
            comm_world,
            crystal,
            [[wfn_copy]],
            dt,
            numsteps,
            gamma_efield_kick,
            "z",
            write_freq=-1,
        )
    )
    if comm_world.rank == 0:
        final = dip_z[-1, 2].real
        err = abs(final - REF_FINAL_DIPZ[exp_method])
        assert err < 1e-4, (
            f"{exp_method} under MPI (np={comm_world.size}) diverged from "
            f"serial reference: got {final!r}, expected "
            f"{REF_FINAL_DIPZ[exp_method]!r} (err={err:.3e})"
        )
        print(
            f"OK: np={comm_world.size}, {exp_method} err vs serial reference = {err:.3e}"
        )
