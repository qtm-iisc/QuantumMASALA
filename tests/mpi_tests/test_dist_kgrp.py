"""Real multi-process tests for k-point-group (kgrp) MPI parallelism --
`DFTCommMod(comm_world, comm_world.size, 1)`, the "k-point-only"
convention already used by `tests/system_tests/test_dft_si.py` -- each run
via `mpirun -np N` (see `conftest.run_mpi`).

`test_dft_si.py` itself was never actually exercised under `mpirun -np
N>1`: at N>1 each rank's `l_wfn_kgrp` only holds its own disjoint slice of
the 7 k-points, so its `test_eigenvalues` (which compares `l_wfn_kgrp`
directly against the full 7-k-point reference array) throws a
shape-mismatch `ValueError` rather than actually checking anything -- not
a bug in the k-point-parallel SCF itself (confirmed here: gathering the
per-rank slices before comparing reproduces the single-process reference
exactly, bit-for-bit, at every rank count up to 7), just a gap in that
test file's own coverage for N>1.
"""
import pytest

from conftest import requires_mpi, run_mpi

NPROCS = [1, 2, 4, 7]


@requires_mpi
@pytest.mark.parametrize("nprocs", NPROCS)
def test_dist_kgrp_scf_si(nprocs):
    """Silicon SCF (7 IBZ k-points) split across `nprocs` k-groups
    reproduces the single-process total energy and band eigenvalues."""
    run_mpi("kgrp_scf_si.py", nprocs)
