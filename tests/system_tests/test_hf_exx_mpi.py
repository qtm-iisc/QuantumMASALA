import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI

from qtm.constants import RYDBERG
from qtm.dft import DFTCommMod, KSWfn
from qtm.dft.exchange import MultiKExchangeDriver
from qtm.gspace import GSpace, GkSpace
from qtm.lattice import RealLattice, ReciLattice
from qtm.mpi import QTMComm


def _build_test_system():
    """Construct a tiny artificial four-k-point system."""

    reallat = RealLattice.from_alat(
        8.0,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )

    recilat = ReciLattice.from_reallat(reallat)

    gwfn = GSpace(
        recilat,
        8.0 * RYDBERG,
    )

    kpoints = [
        (0.00, 0.00, 0.00),
        (0.25, 0.00, 0.00),
        (0.00, 0.25, 0.00),
        (0.00, 0.00, 0.25),
    ]

    return reallat, gwfn, kpoints


def _make_kswfn(gwfn, kpoint, ik, weight):
    """Create one deterministic occupied Kohn-Sham wavefunction."""

    gkspc = GkSpace(
        gwfn,
        kpoint,
    )

    kswfn = KSWfn(
        gkspc,
        weight,
        1,
        False,
    )

    ncoeff = kswfn.evc_gk.data.size

    x = np.arange(
        1,
        ncoeff + 1,
        dtype=float,
    )

    coeff = (
        (ik + 1.0) * x
        + 1j * (ik + 2.0) / x
    )

    kswfn.evc_gk.data[:] = coeff.reshape(
        kswfn.evc_gk.data.shape
    )

    kswfn.evc_gk.normalize()

    kswfn.occ[:] = 1.0

    return kswfn


def test_multik_fock_mpi_matches_serial():
    """MPI k-point EXX must reproduce the serial full Fock operator."""

    comm_world = QTMComm(MPI.COMM_WORLD)

    # This regression test is specifically designed for:
    #
    #     4 k-points -> 4 k-groups -> 1 MPI rank per k-group
    #
    # Normal single-process pytest runs should therefore skip it.
    if comm_world.size != 4:
        pytest.skip(
            "MPI EXX regression test requires exactly 4 MPI ranks"
        )

    dftcomm = DFTCommMod(
        comm_world,
        n_kgrp=comm_world.size,
        pwgrp_size=1,
    )

    reallat, gwfn, kpoints = _build_test_system()

    nk = len(kpoints)
    weight = 1.0 / nk

    # Construct the same deterministic four wavefunctions on every rank.
    #
    # This gives us an independent serial reference while the MPI driver
    # receives only the k-point owned by the current rank.
    all_kswfns = [
        _make_kswfn(
            gwfn,
            kpoints[ik],
            ik,
            weight,
        )
        for ik in range(nk)
    ]

    local_kswfn = all_kswfns[comm_world.rank]

    # --------------------------------------------------------
    # Serial reference
    # --------------------------------------------------------

    serial_driver = MultiKExchangeDriver(
        volume=float(reallat.cellvol),
        nk_total=nk,
        exxdiv_treatment="none",
        exx_Rc=None,
        occ_spin_factor=1.0,
        kweight_bz_factor=1.0,
        use_ace=False,
        kroot_comm=None,
    )

    serial_driver.update_from_kswfns(
        all_kswfns,
        build_ace=False,
    )

    serial_block = serial_driver.block_for_kswfn(
        local_kswfn
    )

    vx_serial = serial_block.apply_fock(
        local_kswfn.evc_gk[0]
    )

    # --------------------------------------------------------
    # MPI k-point-distributed calculation
    # --------------------------------------------------------

    mpi_driver = MultiKExchangeDriver(
        volume=float(reallat.cellvol),
        nk_total=nk,
        exxdiv_treatment="none",
        exx_Rc=None,
        occ_spin_factor=1.0,
        kweight_bz_factor=1.0,
        use_ace=False,
        kroot_comm=dftcomm.kroot_intra,
    )

    mpi_driver.update_from_kswfns(
        [local_kswfn],
        build_ace=False,
    )

    mpi_block = mpi_driver.block_for_kswfn(
        local_kswfn
    )

    vx_mpi = mpi_block.apply_fock(
        local_kswfn.evc_gk[0]
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert np.allclose(
        vx_mpi.data,
        vx_serial.data,
        rtol=1e-11,
        atol=1e-12,
    )
