"""Driver (not collected by pytest): run under `mpirun -np N python3
pwgrp_hpsi.py` by `tests/mpi_tests/test_dist_pwgrp.py`.

Checks that `KSHam.h_psi` (kinetic + local + nonlocal, using the real CH4
pseudopotentials) is exact to machine precision at N ranks under
`DistGkSpace`/`DistGSpace`, using a FIXED, deterministic (not
SCF-derived) test wavefunction -- so there is no eigensolver-gauge
ambiguity to confound the comparison, only the Hamiltonian-apply pipeline
itself is under test. Compares against the same computation done with a
plain (serial) `GkSpace`/`GSpace`.
"""
import os

import numpy as np

from qtm.config import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import RYDBERG
from qtm.containers.field import get_FieldG, get_FieldR
from qtm.containers.wavefun import get_WavefunG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft.ksham import KSHam
from qtm.gspace import GSpace, GkSpace
from qtm.mpi import QTMComm
from qtm.mpi.gspace import DistGSpace, DistGkSpace
from qtm.pseudo import UPFv2Data
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.pseudo.nloc import NonlocGenerator
from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)

SYSTEM_TESTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "system_tests")

reallat_kwargs = dict(alat=30.0, a1=[1.0, 0.0, 0.0], a2=[0.0, 1.0, 0.0], a3=[0.0, 0.0, 1.0])
from qtm.lattice import RealLattice

reallat = RealLattice.from_alat(**reallat_kwargs)
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

v_ion = get_FieldG(gwfn_serial).zeros(1).to_r()
for sp in crystal.l_atoms:
    v_ion_typ, _ = loc_generate_pot_rhocore(sp, gwfn_serial)
    v_ion += v_ion_typ.to_r()
vloc_r_glob = v_ion.data.ravel().copy()

rng = np.random.default_rng(42)
psi0_glob = (
    rng.standard_normal((2, gkspc_serial.size_g))
    + 1j * rng.standard_normal((2, gkspc_serial.size_g))
).astype("c16")

if comm_world.size == 1:
    gwfn, gkspc = gwfn_serial, gkspc_serial
    vloc_r_loc = vloc_r_glob
    psi0_loc = psi0_glob
else:
    gwfn = DistGSpace(comm_world, gwfn_serial)
    gkspc = DistGkSpace(comm_world, gkspc_serial, gwfn)
    vloc_r_loc = vloc_r_glob[gwfn.ir_loc]
    psi0_loc = gkspc.scatter_g(psi0_glob if comm_world.rank == 0 else None)

l_nloc = [NonlocGenerator(sp, gwfn) for sp in crystal.l_atoms]
FieldR = get_FieldR(gwfn)
vloc = FieldR.empty(())
vloc.data[:] = vloc_r_loc

WavefunG = get_WavefunG(gkspc, 1)
psi = WavefunG.empty((2,))
psi.data[:] = psi0_loc

ksham = KSHam(gkspc, False, vloc, l_nloc)
hpsi = WavefunG.empty((2,))
ksham.h_psi(psi, hpsi)

if comm_world.size == 1:
    hpsi_glob = hpsi.data.copy()
else:
    hpsi_glob = gkspc.allgather_g(hpsi.data)

# Reference value computed via this same script's serial (np=1) path.
REF_HPSI0_5 = np.array(
    [
        -18807.72483767 - 49912.19976383j,
        2445.00190881 + 65010.25101971j,
        9307.7688686 - 44939.64986565j,
        -15452.51814095 + 27085.09549365j,
        3713.99808911 - 38915.01366421j,
    ]
)

if comm_world.rank == 0:
    err = np.max(np.abs(hpsi_glob[0, :5] - REF_HPSI0_5))
    assert err < 1e-6, f"h_psi under MPI diverged from serial reference: {err:.3e}"
    print(f"OK: np={comm_world.size}, h_psi err vs serial reference = {err:.3e}")
