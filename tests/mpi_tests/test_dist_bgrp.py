"""Real multi-process tests for band-group (bgrp) MPI parallelism --
`DFTCommMod(comm_world, 1, pwgrp_size)` derives `n_bgrp = n_pwgrp` (since
`n_kgrp=1` here) whenever `comm_world.size > pwgrp_size`, splitting
`numbnd` bands across `n_bgrp` groups within Davidson (see
`qtm.dft.eigsolve.davidson`'s use of `dftcomm.pwgrp_inter_kgrp`) -- a third
parallelization axis, alongside pwgrp (`test_dist_pwgrp.py`) and kgrp
(`test_dist_kgrp.py`), each run via `mpirun -np N` (see `conftest.run_mpi`).

Two `(nprocs, pwgrp_size)` combinations are covered: pure band-group
parallelism (`pwgrp_size=1`, so all of `nprocs` splits only across bands)
and band-group + plane-wave-group combined (`pwgrp_size=2`, so bands AND
G-vectors are both split). Both require building the distributed G-space
on `dftcomm.pwgrp_intra` rather than `comm_world`: the two only coincide
when `n_bgrp=1`, so passing `comm_world` (as `test_dist_pwgrp.py` and
`examples/tddft-ch4/ch4_scf_dipz.py` do, where `n_bgrp` is always 1) spans
multiple separate plane-wave groups once real band groups are active, and
crashes with `MPI_ERR_TRUNCATE`.
"""
import pytest

from conftest import requires_mpi, run_mpi

# (nprocs, pwgrp_size) -- n_bgrp = nprocs // pwgrp_size
COMBOS = [(2, 1), (4, 2)]


@requires_mpi
@pytest.mark.parametrize("nprocs,pwgrp_size", COMBOS)
def test_dist_bgrp_scf(nprocs, pwgrp_size):
    """CH4 SCF with band groups active reproduces the serial total
    energy."""
    run_mpi("bgrp_scf_ch4.py", nprocs, args=[pwgrp_size])


@requires_mpi
@pytest.mark.parametrize("nprocs,pwgrp_size", COMBOS)
def test_dist_bgrp_tddft(nprocs, pwgrp_size):
    """A short CH4 TaylorExp dipole-response run, from a band-group-parallel
    SCF ground state, reproduces the serial reference."""
    run_mpi("bgrp_tddft_short.py", nprocs, args=[pwgrp_size], timeout=90.0)
