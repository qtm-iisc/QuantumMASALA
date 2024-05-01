from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal
__all__ = ['KList', 'gen_monkhorst_pack_grid']

import numpy as np
from spglib import get_stabilized_reciprocal_mesh
from qtm.lattice import ReciLattice
from qtm.crystal import Crystal
from qtm.mpi.utils import scatter_slice

from qtm.msg_format import *
from qtm.config import NDArray


def _sanitize_weights(weights: list[float]):
    weights = np.array(weights, dtype='f8')
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    weights /= np.sum(weights)
    return weights


class KList:

    def __init__(self, recilat: ReciLattice, k_coords: np.ndarray,
                 k_weights: np.ndarray,
                 coords_typ: Literal['cryst', 'cart', 'tpiba'] = 'cryst'):
        if not isinstance(recilat, ReciLattice):
            raise TypeError(f"'recilat' must be a {ReciLattice} instance. "
                            f"got '{type(recilat)}'.")
        self.recilat = recilat

        try:
            k_coords = np.asarray(k_coords, dtype='f8', like=self.recilat.recvec)
        except Exception as e:
            raise TypeError(
                type_mismatch_msg('k_coords', k_coords, 'array-like object')
            ) from e
        if k_coords.ndim != 2:
            raise ValueError(
                value_mismatch_msg('k_coords.ndim', k_coords.ndim, 2)
            )
        if k_coords.shape[0] != 3:
            raise ValueError(
                value_mismatch_msg('k_coords.shape[0]', k_coords.shape[0], 3)
            )
        if coords_typ not in ['cryst', 'cart', 'tpiba']:
            raise ValueError(value_not_in_list_msg(
                'coords_typ', coords_typ, ['cryst', 'cart', 'tpiba'])
            )
        if coords_typ == 'cryst':
            k_cryst = k_coords
        elif coords_typ == 'cart':
            k_cryst = self.recilat.cart2cryst(k_coords)
        else:  # if coords_typ == 'tpiba':
            k_cryst = self.recilat.tpiba2cryst(k_coords)
        self.k_cryst = k_cryst

        try:
            k_weights = np.asarray(k_weights, like=self.recilat.recvec)
        except Exception as e:
            raise TypeError(
                type_mismatch_msg('k_weights', k_weights, 'an array-like object')
            ) from e
        if k_weights.shape != (self.numkpts, ):
            raise ValueError(value_mismatch_msg(
                'k_weights.shape', k_weights.shape,
                f'(k_coords.shape[1], ) = ({k_coords.shape[1]}, )'
            ))
        self.k_weights = k_weights

    @property
    def numkpts(self) -> int:
        return self.k_cryst.shape[1]

    @classmethod
    def gamma(cls, recilat: ReciLattice):
        k_cryst = np.zeros((3, 1), dtype='f8')
        k_weights = np.ones(1, dtype='f8')
        return cls(recilat, k_cryst, k_weights)

    @property
    def k_cart(self) -> NDArray:
        return self.recilat.cryst2cart(self.k_cryst, axis=0)

    @property
    def k_tpiba(self) -> NDArray:
        return self.recilat.cryst2tpiba(self.k_cryst, axis=1)

    def __len__(self):
        return self.numkpts

    def __getitem__(self, item) -> \
            KList | tuple[tuple[float, float, float], float]:
        k_cryst = self.k_cryst[:, item]
        weights = self.k_weights[item]
        if weights.ndim == 0:
            return tuple(k_cryst.tolist()), weights
        return KList(self.recilat, k_cryst, weights)

    def scatter(self, n_kgrp: int, i_kgrp: int) -> KList:
        if not isinstance(n_kgrp, int) or n_kgrp <= 0:
            raise TypeError(type_mismatch_msg(
                'n_kgrp', n_kgrp, 'a positive integer'
            ))
        if not isinstance(i_kgrp, int) or not (0 <= i_kgrp < n_kgrp):
            raise TypeError(type_mismatch_msg(
                'i_kgrp', i_kgrp, f"a positive integer less than n_kgrp = {n_kgrp}"
            ))

        sl_kgrp = scatter_slice(self.numkpts, n_kgrp, i_kgrp)
        return self[sl_kgrp]
    
    def __repr__(self) -> str:
        return f"KList(\n\tnumkpts={self.numkpts},\n\trecilat={self.recilat}, \n\tk_cryst={self.k_cryst.T}, \n\tk_weights={self.k_weights})"


def gen_monkhorst_pack_grid(
        crystal: Crystal, grid_shape: tuple[int, int, int],
        shifts: tuple[bool, bool, bool], use_symm: bool = True,
        is_time_reversal: bool = True
) -> KList:
    if not isinstance(crystal, Crystal):
        raise TypeError(
            type_mismatch_msg('crystal', Crystal, Crystal)
        )
    if (not all(isinstance(ni, int) and ni > 0 for ni in grid_shape)
            or len(grid_shape) != 3):
        raise TypeError(
            type_mismatch_seq_msg('grid_shape', grid_shape, 'three positive integers')
        )
    if not all(isinstance(si, bool) for si in shifts) or len(shifts) != 3:
        raise TypeError(
            type_mismatch_msg('shifts', shifts, 'three booleans')
        )
    if not isinstance(is_time_reversal, bool):
        raise TypeError('is_time_reversal', is_time_reversal, bool)
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
                (np.arange(ni // 2 + 1) + 0.5 * si) / ni
                for ni, si in zip(grid_shape, shifts)
            ]
        else:
            ki = [
                (np.arange(-ni // 2 + 1, ni // 2 + 1) + 0.5 * si) / ni
                for ni, si in zip(grid_shape, shifts)
            ]
        k_mesh_cryst = np.meshgrid(*ki, indexing="ij")
        k_cryst = np.transpose([np.ravel(arr) for arr in k_mesh_cryst])
        numk = k_cryst.shape[0]
        weights = np.ones(numk) / numk

    weights = _sanitize_weights(weights)
    return KList(crystal.recilat, k_cryst.T, weights)
