"""Real multi-process tests for plane-wave-group (pwgrp) MPI parallelism
-- `qtm.mpi.gspace.DistGSpace`/`DistGkSpace`, `DFTCommMod(..., pwgrp_size >
1)` -- each run via `mpirun -np N` (see `conftest.run_mpi`), unlike every
other test in the suite, which only ever exercises `MPI4PY_INSTALLED` code
paths within a single process.

`SplitOper` and `CrankNicolson`'s per-band BiCGSTAB solve deadlocks under
`pwgrp_size > 1` unless it uses the distribution-aware solver in
`qtm.tddft_gamma.expoper._bicgstab` -- a bug that only running under more
than one process (which no other test in the suite does) can catch; see
`test_dist_tddft_propagators_no_deadlock` below.
"""
import pytest

from conftest import requires_mpi, run_mpi

NPROCS = [1, 2, 4]


@requires_mpi
@pytest.mark.parametrize("nprocs", NPROCS)
def test_dist_fft_roundtrip(nprocs):
    """`DistGSpace`'s distributed pencil-FFT round trip matches a plain
    serial `GSpace`'s to machine precision."""
    run_mpi("pwgrp_fft_roundtrip.py", nprocs)


@requires_mpi
@pytest.mark.parametrize("nprocs", NPROCS)
def test_dist_hpsi(nprocs):
    """`KSHam.h_psi` (kinetic + local + nonlocal, real CH4 pseudopotentials)
    matches the serial result to machine precision under `DistGkSpace`."""
    run_mpi("pwgrp_hpsi.py", nprocs)


@requires_mpi
@pytest.mark.parametrize("nprocs", NPROCS)
def test_dist_compute_rho(nprocs):
    """`KSWfn.compute_rho` matches the serial result to machine precision
    under `DistGkSpace`."""
    run_mpi("pwgrp_rho.py", nprocs)


@requires_mpi
@pytest.mark.parametrize("nprocs", NPROCS)
def test_dist_tddft_propagators_no_deadlock(nprocs):
    """A short CH4 dipole-response run with TaylorExp, SplitOper and
    CrankNicolson all complete (regression guard for the BiCGSTAB-under-MPI
    deadlock) and agree with the serial reference under `pwgrp_size>1`.
    `run_mpi`'s timeout turns a reintroduced deadlock into a prompt
    failure rather than a hang."""
    run_mpi("pwgrp_tddft_short.py", nprocs, timeout=90.0)
