"""Driver (not collected by pytest): run under `mpirun -np N python3
kgrp_scf_si.py` by `tests/mpi_tests/test_dist_kgrp.py`.

Checks k-point-group (kgrp) MPI parallelism for SCF -- a different
parallelization axis than the plane-wave-group (pwgrp) distribution
exercised by `tests/mpi_tests/scripts/pwgrp_*.py` -- against the same
silicon system and reference values as `tests/system_tests/test_dft_si.py`
(computed there at a single process). `DFTCommMod(comm_world,
comm_world.size, 1)` assigns each rank its own size-1 k-group holding a
DISJOINT SLICE of the k-points (`l_wfn_kgrp` only has the rank's local
k-points, not all of them) -- `test_dft_si.py` itself was never actually
exercised under `mpirun -np N>1` and its `test_eigenvalues` compares
`l_wfn_kgrp` directly against the FULL reference array, which only has as
many rows as this rank's local k-point slice once N>1 (confirmed: this
throws a shape-mismatch `ValueError`, e.g. (4,4) vs (7,4) at N=2 -- not a
bug in the underlying k-point-parallel SCF, just a gap in that test file's
coverage). This script fixes that by gathering each rank's local
eigenvalues (via a plain `allgather`, valid here since kgrp_size=1 under
this "k-point-only" `DFTCommMod` convention, so each rank is exactly one
k-group and rank order matches k-group order) before comparing to the
full reference.
"""
import os

import numpy as np

from qtm.config import qtmconfig

qtmconfig.set_gpu(False)

from qtm.constants import RYDBERG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, scf
from qtm.gspace import GSpace
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.lattice import RealLattice
from qtm.mpi import QTMComm
from qtm.pseudo import UPFv2Data
from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)  # k-point-only parallelization

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
assert kpts.numkpts == 7, f"expected 7 IBZ k-points, got {kpts.numkpts}"

ecut_wfn = 5 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
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

# Reference values from tests/system_tests/test_dft_si.py (computed there
# at a single process; kgrp parallelism must reproduce them exactly, since
# unlike the pwgrp/Davidson case, k-points don't share a degenerate
# eigenspace across each other -- there is no gauge ambiguity here).
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
# kgrp_size=1 under this DFTCommMod convention (n_bgrp=1, pwgrp_size=1), so
# every rank IS its own k-group and rank order matches k-group (hence
# k-point-block) order -- a plain allgather reassembles the full,
# correctly-ordered k-point list.
all_evl = np.concatenate(comm_world.allgather(local_evl), axis=0)

if comm_world.rank == 0:
    en_err = abs(en.total - REF_TOTAL_ENERGY)
    evl_err = np.max(np.abs(all_evl - REF_EIGENVALUES))
    assert en_err < 1e-5, f"kgrp SCF total energy diverged: err={en_err:.3e}"
    assert evl_err < 1e-4, f"kgrp SCF eigenvalues diverged: err={evl_err:.3e}"
    print(
        f"OK: np={comm_world.size} (n_kgrp={dftcomm.n_kgrp}), "
        f"energy err={en_err:.3e}, eigenvalue err={evl_err:.3e}"
    )
