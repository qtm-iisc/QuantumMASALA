from __future__ import annotations

__all__ = [
    "check_lattice",
    "check_basisatoms",
    "check_crystal",
    "check_gspace",
    "check_gkspace",
    "check_kpts",
    "check_system",
]

import numpy as np
from qtm.lattice import Lattice, RealLattice, ReciLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.gspace import GSpace, GkSpace
from qtm.kpts import KList

from .comm import QTMComm

from qtm.config import MPI4PY_INSTALLED
from qtm.msg_format import obj_mismatch_msg

EPS = 1e-8


def mismatch_msg(obj_name: str, obj_type: type):
    return (
        f"'{obj_type}' instance '{obj_name}' is not equivalent across all "
        f"processes. Refer to the above messages for more info."
    )


def check_lattice(comm: QTMComm, lattice: Lattice):
    assert isinstance(comm, QTMComm)
    assert isinstance(lattice, Lattice)
    try:
        primvec_root = lattice.primvec.copy()
        comm.Bcast(primvec_root)
        assert np.allclose(lattice.primvec, primvec_root)
        if isinstance(lattice, RealLattice):
            assert abs(lattice.alat - comm.bcast(lattice.alat)) <= EPS
        elif isinstance(lattice, ReciLattice):
            assert abs(lattice.tpiba - comm.bcast(lattice.tpiba)) <= EPS
    except AssertionError as e:
        raise Exception(mismatch_msg("lattice", Lattice)) from e


def check_basisatoms(comm: QTMComm, atoms: BasisAtoms, check_reallat: bool = False):
    assert isinstance(comm, QTMComm)
    assert isinstance(atoms, BasisAtoms)
    assert isinstance(check_reallat, bool)
    check_reallat = comm.bcast(check_reallat)
    try:
        assert atoms.label == comm.bcast(atoms.label)
        if check_reallat:
            check_lattice(comm, atoms.reallat)
        assert atoms.valence == comm.bcast(atoms.valence)
        assert atoms.ppdata.filename == comm.bcast(atoms.ppdata.filename)
        assert atoms.ppdata.md5_checksum == comm.bcast(atoms.ppdata.md5_checksum)
        assert atoms.mass == comm.bcast(atoms.mass) if atoms.mass is not None else True
        r_cryst_root = atoms.r_cryst.copy()
        comm.Bcast(r_cryst_root)
        assert np.allclose(atoms.r_cryst, r_cryst_root)
    except AssertionError as e:
        raise Exception(mismatch_msg("atoms", BasisAtoms)) from e


def check_crystal(comm: QTMComm, crystal: Crystal):
    assert isinstance(comm, QTMComm)
    assert isinstance(crystal, Crystal)
    try:
        check_lattice(comm, crystal.reallat)
        for typ in crystal.l_atoms:
            check_basisatoms(comm, typ)
    except AssertionError as e:
        raise Exception(mismatch_msg("crystal", Crystal)) from e


def check_gspace(comm: QTMComm, gspc: GSpace):
    assert isinstance(comm, QTMComm)
    assert isinstance(gspc, GSpace)
    try:
        assert np.allclose(gspc.recilat.recvec, comm.bcast(gspc.recilat.recvec))
        assert abs(gspc.ecut - comm.bcast(gspc.ecut)) <= 1e-8
        for ipol in range(3):
            assert gspc.grid_shape[ipol] == comm.bcast(gspc.grid_shape[ipol])
        is_dist = False
        if MPI4PY_INSTALLED:
            from .gspace import DistGSpace

            if isinstance(gspc, DistGSpace):
                is_dist = True
        assert is_dist == comm.bcast(is_dist), (
            f"'gspc' at root is a "
            f"{'parallel' if not is_dist else 'serial'} instance while "
            f"the local instance is {'parallel' if is_dist else 'serial'}."
        )
    except AssertionError as e:
        raise Exception(mismatch_msg("gspc", GSpace)) from e


def check_gkspace(comm: QTMComm, gkspc: GkSpace, check_gwfn: bool = False):
    assert isinstance(comm, QTMComm)
    assert isinstance(gkspc, GkSpace)
    assert isinstance(check_gwfn, bool)
    check_gwfn = comm.bcast(check_gwfn)
    try:
        if check_gwfn:
            check_gspace(comm, gkspc.gwfn)
        for ipol in range(3):
            assert gkspc.k_cryst[ipol] == comm.bcast(gkspc.k_cryst[ipol])
        is_dist = False
        if MPI4PY_INSTALLED:
            from .gspace import DistGkSpace

            if isinstance(gkspc, DistGkSpace):
                is_dist = True
        assert is_dist == comm.bcast(is_dist), (
            f"'gkspc' at root is a "
            f"{'parallel' if not is_dist else 'serial'} instance while "
            f"the local instance is {'parallel' if is_dist else 'serial'}."
        )
    except AssertionError as e:
        raise Exception(mismatch_msg("gkspc", gkspc)) from e


def check_kpts(comm: QTMComm, kpts: KList, check_recilat: bool = False):
    assert isinstance(comm, QTMComm)
    assert isinstance(kpts, KList)
    assert isinstance(check_recilat, bool)
    check_recilat = comm.bcast(check_recilat)
    try:
        if check_recilat:
            check_lattice(comm, kpts.recilat)
        k_cryst_root = kpts.k_cryst.copy()
        comm.Bcast(k_cryst_root)
        assert np.allclose(kpts.k_cryst, k_cryst_root)
        k_weights_root = kpts.k_weights.copy()
        comm.Bcast(k_weights_root)
        assert np.allclose(kpts.k_weights, k_weights_root)
    except AssertionError as e:
        raise Exception(mismatch_msg("kpts", kpts)) from e


def check_system(
    comm: QTMComm, crystal: Crystal, grho: GSpace, gwfn: GSpace, kpts: KList
):
    assert isinstance(comm, QTMComm)
    with comm:
        check_crystal(comm, crystal)
        check_gspace(comm, grho)
        assert grho.recilat == crystal.recilat, obj_mismatch_msg(
            "grho.recilat", grho.recilat, "crystal.recilat", crystal.recilat
        )
        check_gspace(comm, gwfn)
        assert gwfn.recilat == grho.recilat, obj_mismatch_msg(
            "gwfn.recilat", gwfn.recilat, "crystal.recilat", crystal.recilat
        )
        check_kpts(comm, kpts)
        assert kpts.recilat == crystal.recilat, obj_mismatch_msg(
            "kpts.recilat", kpts.recilat, "crystal.recilat", crystal.recilat
        )
