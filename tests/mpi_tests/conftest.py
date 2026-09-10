"""Shared helper for the real multi-process MPI tests in this directory.

Everywhere else in the test suite, `MPI4PY_INSTALLED` communicators are
exercised with exactly one process (whatever invoked pytest itself) --
`test_comm.py`'s own docstring notes that the real cross-process
distributed data paths in `qtm.mpi.gspace`/`qtm.mpi.containers` are
"meaningless to unit test" that way. The tests in this directory instead
shell out to `mpirun -np N python3 <driver script>` for each of several
`N`, so the distributed code actually runs across real, separate
processes -- each driver script (in `scripts/`) does its own comparison
against a known-good serial reference and raises `AssertionError` (a
nonzero exit code) on mismatch.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qtm.config import MPI4PY_INSTALLED

SCRIPTS_DIR = Path(__file__).parent / "scripts"

MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")

requires_mpi = pytest.mark.skipif(
    not MPI4PY_INSTALLED or MPIRUN is None,
    reason="mpi4py and/or an mpirun/mpiexec launcher is not available",
)


def run_mpi(script_name: str, nprocs: int, args=(), timeout: float = 120.0):
    """Runs `scripts/<script_name>` under `mpirun -np nprocs`, returning
    the completed-process result. A nonzero exit code (an `AssertionError`
    in the driver script, or a genuine crash) fails the calling test, with
    the subprocess's captured output attached for debugging. `timeout`
    turns a reintroduced deadlock (see `pwgrp_tddft_short.py`) into a
    prompt test failure instead of a hung test run -- but note a killed
    mpirun job can leave stale PRTE/shared-memory state behind that
    degrades or hangs the *next* `mpirun` invocation on the same machine
    even though the code itself is fine; if a test times out, re-run it
    alone before concluding the underlying code regressed.
    """
    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run(
        [MPIRUN, "-np", str(nprocs), sys.executable, str(script_path), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"{script_name} failed under mpirun -np {nprocs} "
        f"(exit code {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return result
