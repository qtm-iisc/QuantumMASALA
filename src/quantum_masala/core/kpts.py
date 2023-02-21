from __future__ import annotations

__all__ = ["KList", "kpts_distribute"]
from typing import Union, Optional
import numpy as np
from spglib import get_stabilized_reciprocal_mesh

from quantum_masala import config
from quantum_masala.core import ReciprocalLattice, Crystal


def _sanitize_weights(weights: list[float]):
    weights = np.array(weights, dtype='f8')
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    weights /= np.sum(weights)
    return weights


class KList:

    def __init__(self, recilat: ReciprocalLattice, cryst: np.ndarray,
                 weights: np.ndarray):
        self.recilat = recilat
        if cryst.shape[0] != weights.shape[0]:
            raise ValueError("length of 'cryst' and 'weights' do not match. "
                             f"len(cryst)={len(cryst)}, len(weights)={len(weights)}")
        self.len_ = cryst.shape[0]
        if weights.ndim != 1:
            raise ValueError(f"'weights' must be a 1d array. got {weights.ndim}")
        if cryst.ndim != 2 or cryst.shape[1] != 3:
            raise ValueError("'cryst' must be a 2D numpy array representing"
                             "the list of k-points in crystal coordinates"
                             f"located in 3D space. got cryst.shape={cryst.shape}")

        self.cryst = np.empty((self.len_, 3), dtype='f8')
        self.weights = np.empty(self.len_, dtype='f8')
        self.cryst[:] = cryst
        self.weights[:] = weights

    def __len__(self):
        return self.len_

    @property
    def cart(self):
        return self.recilat.cryst2cart(self.cryst, axis=1)

    @property
    def tpiba(self):
        return self.recilat.cryst2tpiba(self.cryst, axis=1)

    def __getitem__(self, item) -> Union[KList,
                                         tuple[tuple[float, float, float], float]]:
        if isinstance(item, slice):
            cryst = self.cryst[item]
            weights = self.weights[item]
            return KList(self.recilat, cryst, weights)
        elif isinstance(item, int):
            return tuple(self.cryst[item]), self.weights[item]
        else:
            raise TypeError(f"indices must be integers or slices, not {type(item)}")

    @classmethod
    def from_cart(cls, crystal, *l_kpts_cart):
        weights, cart = [], []
        for k_cart, k_weight, in l_kpts_cart:
            cart.append(k_cart)
            weights.append(k_weight)

        recilat = crystal.recilat
        cryst = recilat.cart2cryst(cart, axis=1)
        weights = _sanitize_weights(weights)
        return cls(recilat, cryst, weights)

    @classmethod
    def from_cryst(cls, crystal, *l_kpts_cryst):
        weights, cryst = [], []
        for k_cart, k_weight in l_kpts_cryst:
            cryst.append(k_cart)
            weights.append(k_weight)

        recilat = crystal.recilat
        cryst = np.array(cryst)
        weights = _sanitize_weights(weights)
        return cls(recilat, cryst, weights)

    @classmethod
    def from_tpiba(cls, crystal, *l_kpts_tpiba):
        weights, tpiba = [], []
        for k_cart, k_weight in l_kpts_tpiba:
            tpiba.append(k_cart)
            weights.append(k_weight)

        recilat = crystal.recilat
        cryst = recilat.tpiba2cryst(tpiba, axis=1)
        weights = _sanitize_weights(weights)
        return cls(recilat, cryst, weights)

    @classmethod
    def gamma(cls, crystal: Crystal):
        return cls(crystal.recilat,
                   np.zeros((1, 3), dtype='f8'),
                   np.ones(1, dtype='f8'))

    @classmethod
    def mpgrid(cls, crystal: Crystal,
               grid_shape: tuple[int, int, int],
               shifts: tuple[bool, bool, bool],
               use_symm: bool = True,
               is_time_reversal: bool = True,
               ):
        if use_symm:
            ir_reciprocal_mesh = get_stabilized_reciprocal_mesh(
                grid_shape,
                crystal.symm.reallat_rot,
                shifts,
                is_time_reversal,
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
        return cls(crystal.recilat, k_cryst, weights)


def kpts_distribute(kpts: KList, round_robin: bool = False,
                    return_indices: bool = True
                    ) -> Union[KList, (KList, list[int])]:
    pwcomm = config.pwcomm
    numkpts = len(kpts)

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
