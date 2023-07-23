"""Utility functions used in 'gspace' submodule

Contains primarily routines to validate the input arguments used in
instantiating ``GSpace`` instances.

"""
# from __future__ import annotations
from typing import Type
from qtm.config import NDArray
__all__ = ['check_shape', 'check_g_cryst',
           'cryst2idxgrid', 'check_g_idxgrid', 'idxgrid2cryst',
           ]

import numpy as np


def check_shape(shape: tuple[int, int, int]) -> None:
    """Checks if the input argument is a tuple of three positive integers.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Tuple of three positive integers.

    Raises
    -------
    ValueError
        Raised if input argument is not a tuple of three positive integers.
    """
    shape = tuple(shape)
    if not (len(shape) == 3 and all(isinstance(ni, int) and ni > 0 for ni in shape)):
        raise ValueError("'shape' must be a tuple containing 3 positive integers. "
                         f"got {shape} (type {type(shape)})")


def check_g_cryst(shape: tuple[int, int, int], g_cryst: NDArray) -> None:
    """Function to validate a given array containing G-vectors in crystal
    coordinates

    The input list of G-vectors `g_cryst` must be an ``'i8'`` array of shape
    ``(3, numg)``. All vectors must lie within the FFT grid of shape `shape`
    i.e ``g_cryst[idim, :]`` must be within the range
    :math:`[-(shape[idim]//2), (shape[idim]+1)//2)`

    Parameters
    ----------
    shape : tuple[int, int, int]
        Dimensions of the FFT grid
    g_cryst : NDArray
        (``(3, -1)``, ``'i8'``) Input list of G-vectors

    Raises
    ------
    ValueError
        Raised if `shape` is not a tuple of three positive integers.
    ValueError
        Raised if `g_cryst` fails to satisfy aforementioned conditions.
        Refer to the message for further details.
    """
    # Validate 'shape' param
    check_shape(shape)
    # Check if 'g_cryst' is an array
    if not isinstance(g_cryst, NDArray):
        raise TypeError(f"'g_cryst' must be a NDArray ({NDArray}). "
                        f"got {type(g_cryst)}")

    # Check for correct shape and dtype
    if not (g_cryst.ndim == 2 and g_cryst.shape[0] == 3
            and g_cryst.dtype == 'i8'):
        raise ValueError(
            "'g_cryst' must be a 2D array of int64 with shape[0] = 3. got:\n"
            f"ndim = {g_cryst.ndim if hasattr(g_cryst, 'ndim') else 'N/A'}, "
            f"shape = {g_cryst.shape if hasattr(g_cryst, 'shape') else 'N/A'}, "
            f"dtype = {g_cryst.dtype if hasattr(g_cryst, 'dtype') else 'N/A'}, "
        )

    # Check if values are within bounds
    oob = False
    for idim, n in enumerate(shape):
        if np.any(g_cryst[idim] < -(n//2)) or np.any(g_cryst[idim] >= (n + 1)//2):
            oob = True
    if oob:
        raise ValueError("'g_cryst'/'idxgrid' has indices outside of the FFT grid "
                         f"of shape {shape}")


def cryst2idxgrid(shape: tuple[int, int, int], g_cryst: NDArray) -> NDArray:
    """Converts input list of G-vectors in crystal coordinates to a 1D array of
    indices that is used to take/put corresponding values from/to a 3D grid
    with dimensions ``shape``

    Parameters
    ----------
    shape : tuple[int, int, int]
        Dimensions of the FFT Grid
    g_cryst : NDArray
        (``(3, -1)``, ``'i8'``) Input list of G-vectors. Refer to
        ``check_g_cryst``
    Returns
    -------
    idxgrid : NDArray
        A (``(g_cryst.shape[0], )``, ``'i8'``) array that can be used to index
        the corresponding values of ``g_cryst`` in a flattened 3D array with
        dimensions ``shape``
    """
    check_g_cryst(shape, g_cryst)
    n1, n2, n3 = shape
    i1, i2, i3 = g_cryst
    idxgrid = n2 * n3 * (i1 + n1 * (i1 < 0)) \
        + n3 * (i2 + n2 * (i2 < 0)) \
        + (i3 + n3 * (i3 < 0))
    return idxgrid


def check_g_idxgrid(shape: tuple[int, int, int], idxgrid: NDArray):
    """Function to validate ``idxgrid``, which is the flattened indices of a
    3D FFT array with dimensions ``shape``

    ``idxgrid`` must be a ``'i8''`` 1D array. All values must be non-negative
    and less than the prodcut of ``shape``. This ensures indices are within
    bounds.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Dimensions of the FFT grid
    idxgrid : NDArray
        (``(-1, )``, ``'i8'``) Input array of flattened indices

    Raises
    ------
    ValueError
        Raised if ``idxgrid`` fails to satisfy aforementioned conditions.
        Refer to the message for further details.
    """
    check_shape(shape)
    if not (isinstance(idxgrid, NDArray) and idxgrid.dtype == 'i8'
            and idxgrid.ndim == 1):
        raise ValueError(
            "'idxgrid' must be a 1D array of 'int64'. got:\n"
            f"ndim = {idxgrid.ndim if hasattr(idxgrid, 'ndim') else 'N/A'}, "
            f"shape = {idxgrid.shape if hasattr(idxgrid, 'shape') else 'N/A'}, "
            f"dtype = {idxgrid.dtype if hasattr(idxgrid, 'dtype') else 'N/A'}, "
        )

    if np.any((idxgrid < 0) + (idxgrid >= np.prod(shape))):
        raise ValueError("'idxgrid' has invalid values. All entries must non-negative"
                         "and less than the product of 'shape'.")


# TODO: This function doesn't do what it claims it does. Fix it.
def idxgrid2cryst(shape: tuple[int, int, int], idxgrid: NDArray) -> NDArray:
    """Inverse routine of ``cryst2idxgrid``

    Parameters
    ----------
    shape : tuple[int, int, int]
        Dimensions of the FFT Grid
    idxgrid : NDArray
        (``(3, -1)``, ``'i8'``) Input list of flattened indices. Refer to
        ``check_g_idxgrid``.
    Returns
    -------
    g_cryst : NDArray
        List of G-vectors corresponding to ``idxgrid``
    """
    check_g_idxgrid(shape, idxgrid)
    n1, n2, n3 = shape
    g_cryst_1 = idxgrid // (n2 * n3)
    g_cryst_2 = idxgrid % (n2 * n3)
    g_cryst_3 = g_cryst_2 % n3
    g_cryst_2 //= n3
    return np.stack((g_cryst_1, g_cryst_2, g_cryst_3))
