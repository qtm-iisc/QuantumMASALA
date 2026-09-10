"""Driver (not collected by pytest): run under `mpirun -np N python3
pwgrp_rho.py` by `tests/mpi_tests/test_dist_pwgrp.py`.

Checks that `KSWfn.compute_rho` (density from a fixed set of occupied
orbitals, real CH4 crystal) is exact to machine precision at N ranks under
`DistGkSpace`, using a fixed, deterministic (not SCF-derived) test
wavefunction -- no eigensolver-gauge ambiguity.
"""
import os

import numpy as np

from qtm.config import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import RYDBERG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft.kswfn import KSWfn
from qtm.gspace import GSpace, GkSpace
from qtm.lattice import RealLattice
from qtm.mpi import QTMComm
from qtm.mpi.gspace import DistGSpace, DistGkSpace
from qtm.pseudo import UPFv2Data
from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)

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

ecut_wfn = 5 * RYDBERG
ecut_rho = 4 * ecut_wfn
gwfn_serial = GSpace(crystal.recilat, ecut_rho)
gkspc_serial = GkSpace(gwfn_serial, (0.0, 0.0, 0.0), ecut_wfn)

numbnd = 4
rng = np.random.default_rng(7)
psi0_glob = (
    rng.standard_normal((numbnd, gkspc_serial.size_g))
    + 1j * rng.standard_normal((numbnd, gkspc_serial.size_g))
).astype("c16")

if comm_world.size == 1:
    gwfn, gkspc = gwfn_serial, gkspc_serial
    psi0_loc = psi0_glob
else:
    gwfn = DistGSpace(comm_world, gwfn_serial)
    gkspc = DistGkSpace(comm_world, gkspc_serial, gwfn)
    psi0_loc = gkspc.scatter_g(psi0_glob if comm_world.rank == 0 else None)

wfn = KSWfn(gkspc, 1.0, numbnd, False)
wfn.evc_gk.data[:] = psi0_loc
wfn.occ[:] = 2.0

rho = wfn.compute_rho(ret_raw=True)

if comm_world.size == 1:
    rho_g_glob = rho.to_g().data.copy()
else:
    rho_g_glob = gwfn.allgather_g(rho.to_g().data)

# Reference computed via this same script's serial (np=1) path.
REF_RHO0_5 = np.array(
    [
        2.70000000e01 + 0.0j,
        -1.19386087e-02 - 0.10148298j,
        1.10455913e-01 + 0.15578429j,
        1.85295478e-01 - 0.14653351j,
        -2.34360106e-02 - 0.06197142j,
    ]
)

if comm_world.rank == 0:
    err = np.max(np.abs(rho_g_glob.ravel()[:5] - REF_RHO0_5))
    assert err < 1e-8, f"compute_rho under MPI diverged from serial reference: {err:.3e}"
    print(f"OK: np={comm_world.size}, compute_rho err vs serial reference = {err:.3e}")
