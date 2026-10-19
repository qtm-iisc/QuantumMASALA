"""Shared building blocks for the MD integrators (NVE, NVE_BEEMAN, Andersen, LAMMPS-coupled NVE)."""
from __future__ import annotations

from dataclasses import dataclass
from sys import version_info

from qtm.config import MPI4PY_INSTALLED
from qtm.dft import KSWfn
from qtm.mpi import QTMComm

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)


@dataclass
class EnergyData:
    total: float = 0.0
    hwf: float = 0.0
    one_el: float = 0.0
    ewald: float = 0.0
    hartree: float = 0.0
    xc: float = 0.0

    fermi: float | None = None
    smear: float | None = None
    internal: float | None = None

    HO_level: float | None = None
    LU_level: float | None = None


if version_info[1] >= 8:
    from typing import Protocol

    class IterPrinter(Protocol):
        def __call__(self, idxiter: int, runtime: float, scf_converged: bool,
                     e_error: float, diago_thr: float, diago_avgiter: float,
                     en: EnergyData) -> None:
            ...

    class WfnInit:
        def __init__(self, pre_existing_wfns: list[KSWfn]):
            self.pre_existing_wfns = pre_existing_wfns

        def __call__(self, ik: int, kswfn: list[KSWfn]) -> None:
            """Initialize wavefunctions using pre-existing wavefunctions.
            For now it only works for spin unpolarised case"""
            assert len(kswfn) == len(self.pre_existing_wfns[ik])
            for i in range(len(kswfn)):
                kswfn[i].evc_gk.data[:] = self.pre_existing_wfns[ik][i].evc_gk.data
                kswfn[i].evl[:] = self.pre_existing_wfns[ik][i].evl[:]
                kswfn[i].occ[:] = self.pre_existing_wfns[ik][i].occ[:]
else:
    IterPrinter = 'IterPrinter'
    WfnInit = 'WfnInit'
