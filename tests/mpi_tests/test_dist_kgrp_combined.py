"""Real multi-process tests for k-point-group (kgrp) parallelism combined
with plane-wave-group (pwgrp) and/or band-group (bgrp) parallelism --
`DFTCommMod`'s communicator splitting composes correctly across all three
axes simultaneously.

Covers even splits of all three axes, and -- the likeliest place for an
off-by-one bug -- an UNEVEN k-point split (Si's 7 IBZ k-points don't
divide evenly by `n_kgrp=3`, giving k-point-group sizes 3/2/2).
"""
import pytest

from conftest import requires_mpi, run_mpi

# (nprocs, n_kgrp, pwgrp_size) -- n_bgrp = (nprocs // pwgrp_size) // n_kgrp
COMBOS = [
    (4, 2, 2),  # kgrp + pwgrp, even split, n_bgrp=1
    (4, 2, 1),  # kgrp + bgrp, n_bgrp=2
    (8, 2, 2),  # all three combined
    (6, 3, 1),  # UNEVEN kgrp split (7 kpts / 3 kgrps = 3,2,2), n_bgrp=2
    (6, 3, 2),  # uneven kgrp split + pwgrp combined
]


@requires_mpi
@pytest.mark.parametrize("nprocs,n_kgrp,pwgrp_size", COMBOS)
def test_dist_kgrp_combined_scf(nprocs, n_kgrp, pwgrp_size):
    """Si SCF (7 IBZ k-points) under combined kgrp/pwgrp/bgrp parallelism
    reproduces the single-process total energy AND full per-k-point
    eigenvalue array."""
    run_mpi("kgrp_combined_scf_si.py", nprocs, args=[n_kgrp, pwgrp_size])
