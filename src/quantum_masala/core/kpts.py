from __future__ import annotations

__all__ = ["KPoints", "kpts_distribute"]
from typing import Union
import numpy as np
from spglib import get_ir_reciprocal_mesh

from quantum_masala import config
from quantum_masala.core import ReciprocalLattice, Crystal


def _sanitize_weights(weights: list[float]):
    weights = np.array(weights, dtype='f8')
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    weights /= np.sum(weights)
    return weights


class KPoints:

    def __init__(self, recilat: ReciprocalLattice, numkpts: int, cryst: np.ndarray,
                 weights: np.ndarray):
        self.recilat = recilat
        self.numk = numkpts
        self.cryst = np.empty((3, self.numk), dtype='f8')
        self.weights = np.empty(self.numk, dtype='f8')
        self.cryst[:] = cryst
        self.weights[:] = weights

    @property
    def cart(self):
        return self.recilat.cryst2cart(self.cryst)

    @property
    def tpiba(self):
        return self.recilat.cryst2tpiba(self.cryst)

    def __getitem__(self, item) -> Union[KPoints,
                                         tuple[tuple[float, float, float], float]]:
        if isinstance(item, slice):
            cryst = self.cryst[:, item]
            weights = self.weights[item]
            numkpts = len(weights)
            return KPoints(self.recilat, numkpts, cryst, weights)
        elif isinstance(item, int):
            return tuple(self.cryst[:, item]), self.weights[item]
        else:
            raise TypeError(f"indices must be integers or slices, not {type(item)}")

    @classmethod
    def from_cart(cls, recilat, *l_kpts_cart):
        weights, cart = [], []
        for k_cart, k_weight, in l_kpts_cart:
            cart.append(k_cart)
            weights.append(k_weight)

        cryst = recilat.cart2cryst(cart, axis=1)
        weights = _sanitize_weights(weights)
        return cls(recilat, len(cryst), cryst.T, weights)

    @classmethod
    def from_cryst(cls, recilat, *l_kpts_cryst):
        weights, cryst = [], []
        for k_cart, k_weight in l_kpts_cryst:
            cryst.append(k_cart)
            weights.append(k_weight)

        cryst = np.array(cryst)
        weights = _sanitize_weights(weights)
        return cls(recilat, len(cryst), cryst.T, weights)

    @classmethod
    def from_tpiba(cls, recilat, *l_kpts_tpiba):
        weights, tpiba = [], []
        for k_cart, k_weight in l_kpts_tpiba:
            tpiba.append(k_cart)
            weights.append(k_weight)

        cryst = recilat.tpiba2cryst(tpiba, axis=1)
        weights = _sanitize_weights(weights)
        return cls(recilat, len(cryst), cryst.T, weights)

    @classmethod
    def gamma(cls, recilat):
        return cls(recilat, 1, np.zeros((3, 1), dtype='f8'), np.ones(1, dtype='f8'))

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
                symprec=config.spglib_symprec,
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

        weights = _sanitize_weights(weights)
        return cls(cryst.recilat, len(k_cryst), k_cryst.T, weights)


def kpts_distribute(kpts: KPoints, round_robin: bool = True,
                    return_indices: bool = True) -> Union[KPoints,
                                                          (KPoints, list[int])]:
    pwcomm = config.pwcomm
    numkpts = kpts.numk

    numkgrp, idxkgrp = pwcomm.numkgrp, pwcomm.idxkgrp
    if numkpts < pwcomm.numkgrp:
        raise ValueError(f"number of k-groups larger than total number of input k-points.\n"
                         f"Please restart with less number of k-groups.\n"
                         f"Got 'numkgrp'={pwcomm.numkgrp}, 'numkpts'={numkpts}")

    if round_robin:
        sl = slice(idxkgrp, None, numkgrp)
    else:
        start = (numkpts // numkgrp) * idxkgrp \
            + min(idxkgrp, numkpts % numkgrp)
        stop = (numkpts // numkgrp) * (idxkgrp + 1) \
            + min(idxkgrp + 1,  numkpts % numkgrp)
        sl = slice(start, stop)

    if return_indices:
        return kpts[sl], list(range(numkpts)[sl])
    else:
        return kpts[sl]