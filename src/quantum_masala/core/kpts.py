
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from spglib import get_ir_reciprocal_mesh

from .cryst import ReciprocalLattice, Crystal
from .mpicomm import PWComm

from quantum_masala.config import SPGLIB_CONFIG
SYMPREC = SPGLIB_CONFIG["SYMPREC"]

ROUND_ROBIN_KPTS = True


class KPoints:
    recilat: ReciprocalLattice
    cryst: np.ndarray
    weights: np.ndarray
    numk: int = field(init=False)

    def __init__(self, recilat: ReciprocalLattice, cryst: ArrayLike,
                 weights: ArrayLike):
        self.recilat = recilat
        self.weights = np.array(weights, dtype='f8')
        self.weights /= np.sum(self.weights)
        self.cryst = np.empty((3, len(self.weights)), dtype='f8')
        self.cryst[:] = cryst
        self.numk = len(self.weights)

    @classmethod
    def from_cart(cls, recilat: ReciprocalLattice, *l_kpts_cart):
        weights = []
        cart = []
        for k_cart, k_weight in l_kpts_cart:
            cart.append(k_cart)
            weights.append(k_weight)

        cryst = recilat.cart2cryst(cart, axis=1).T
        return cls(recilat, cryst, weights)

    @classmethod
    def from_cryst(cls, recilat: ReciprocalLattice, *l_kpts_cryst):
        weights = []
        cryst = []
        for k_cart, k_weight in l_kpts_cryst:
            cryst.append(k_cart)
            weights.append(k_weight)

        cryst = np.array(cryst).T
        return cls(recilat, cryst, weights)

    @classmethod
    def from_tpiba(cls, recilat: ReciprocalLattice, *l_kpts_tpiba):
        weights = []
        tpiba = []
        for k_cart, k_weight in l_kpts_tpiba:
            tpiba.append(k_cart)
            weights.append(k_weight)

        cryst = recilat.tpiba2cryst(tpiba, axis=1).T
        return cls(recilat, cryst, weights)

    @property
    def cart(self):
        return self.recilat.cryst2cart(self.cryst)

    @property
    def tpiba(self):
        return self.recilat.cryst2tpiba(self.cryst)

    def __getitem__(self, x):
        if isinstance(x, int):
            return self.cryst[:, x], self.weights[x]
        else:
            return KPoints(self.recilat, self.cryst[(slice(None), x)], self.weights[x])

    @classmethod
    def mpgrid(cls, cryst: Crystal,
               grid_shape: tuple[int, int, int],
               shifts: tuple[bool, bool, bool],
               use_symm: bool = True,
               is_time_reversal: bool = True):
        if use_symm:
            ir_reciprocal_mesh = get_ir_reciprocal_mesh(
                grid_shape,
                cryst.spglib_cell,
                shifts,
                is_time_reversal,
                symprec=SYMPREC,
            )
            mapping, grid = ir_reciprocal_mesh
            iarr_irred, weights = np.unique(mapping, return_counts=True)
            ir_grid = np.array(grid[iarr_irred])

            k_cryst = (ir_grid + 0.5 * np.array(shifts)) / grid_shape
        else:
            if is_time_reversal:
                ki = [
                    np.arange(ni // 2 + 1) + 0.5 * si / ni
                    for ni, si in zip(grid_shape, shifts)
                ]
            else:
                ki = [
                    np.arange(-ni // 2 + 1, ni // 2 + 1) + 0.5 * si / ni
                    for ni, si in zip(grid_shape, shifts)
                ]
            k_mesh_cryst = np.meshgrid(*ki, indexing="ij")
            k_cryst = np.transpose([np.ravel(arr) for arr in k_mesh_cryst])
            numk = k_cryst.shape[0]
            weights = np.ones(numk) / numk
        return cls(cryst.recilat, k_cryst.T, weights)


def sync_kpts_all(pwcomm: PWComm, kpts_all: KPoints):
    world_comm = pwcomm.world_comm
    if kpts_all.numk != world_comm.bcast(kpts_all.numk):
        raise ValueError("'kpts_all' does not match across all processes.")

    world_comm.Bcast(kpts_all.cryst)
    world_comm.Bcast(kpts_all.weights)
    return kpts_all


class KPointsKgrp(KPoints):
    pwcomm = PWComm()

    def __init__(self, kpts_all: KPoints):
        self.kpts_all = sync_kpts_all(self.pwcomm, kpts_all)

        numk_all = self.kpts_all.numk
        numkgrp = self.pwcomm.numkgrp
        if numk_all < numkgrp:
            raise ValueError(f"number of k-groups larger than total number of input k-points.\n"
                             f"Please restart with less number of k-groups.\n"
                             f"Got 'numkgrp'={numkgrp}, 'numk'={numk_all}")
        idxkgrp = self.pwcomm.idxkgrp
        if ROUND_ROBIN_KPTS:
            l_ikpts = list(range(idxkgrp, numk_all, numkgrp))
        else:
            start = (numk_all // numkgrp) * idxkgrp + min(idxkgrp, numk_all % numkgrp)
            stop = start + (numk_all // numkgrp) + (idxkgrp < numk_all % numkgrp)
            l_ikpts = list(range(start, stop))
        super().__init__(kpts_all.recilat,
                         kpts_all.cryst[:, l_ikpts],
                         kpts_all.weights[l_ikpts]
                         )

    def integrate_kspc(self, arr: np.ndarray, axis_k: int = 0):
        if arr.shape[axis_k] != self.numk:
            raise ValueError(
                f"'arr.shape[axis_k]' must be equal to 'numk'. Expected {self.numk}. "
                f"Got {arr.shape[axis_k]} for 'axis_k'={axis_k}"
            )

        weights = np.expand_dims(
            self.weights, axis=[idim for idim in range(arr.ndim) if idim != axis_k]
        )
        int_k_proc = np.sum(arr * weights, axis=axis_k)
        self.pwcomm.world_comm.Allreduce_sum(int_k_proc)
        return int_k_proc
