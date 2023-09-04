from typing import List
from qtm.lattice import ReciLattice
from qtm.klist import KList

import numpy as np
from numpy.typing import ArrayLike


class QPoints(KList):
    """class QPoints: Stores q-grid

    The core difference between k-grid and q-grid is that q-grid is always gamma-centred.
    Further, to handle divergences associated with qvec=(0,0,0), this class provides an
    attribute ``q0vec``, which can be used instead of (0,0,0), when needed.

    Attributes
    ----------
    q0vec: np.ndarray
        Stores q0 vector; shape=(3,)
    index_q0: int
        index of q0 in the ``cryst`` list provided as argument to constructor
    is_q0: List[bool]
        The fifth column in qpts list provided in epsilon.inp
    cryst: ArrayLike
        (transpose of) qpt list provided in inp files; i.e. _with_ shifted gamma point.
        NOTE: This decision seems OK because energy and Vcoul are computed using shifted gamma point,
        but needs to be checked against BGW's handling of cutoff for q=0.
    """

    def __init__(
        self,
        recilat: ReciLattice,
        is_q0: List[bool],
        cryst: ArrayLike,
    ):
        """Initialize QPoints

        Parameters
        ----------
        recilat : ReciLattice
            Get from GSpace or KList
        cryst : ArrayLike
            List of q-vectors in crystal coordinates, provided in inp files
        is_q0 : List[bool]
            ``is_q0`` column from inp files
        """
        # cryst_gamma = np.array(cryst)

        index_q0 = None
        if is_q0 != None:
            index_q0 = np.where(is_q0)[0][0]
        else:
            index_q0 = np.argmin(recilat.norm2(cryst.T))

        # cryst_gamma[:,index_q0] = 0
        weights = np.ones(cryst.shape[0])
        super().__init__(recilat, cryst, weights)
        self.index_q0 = index_q0
        self.q0vec = cryst[self.index_q0, :]
        # print("q0vec init:", self.q0vec)
        self.numq = cryst.shape[0]
        self.is_q0 = is_q0  # if is_q0!=None else [False]*len(cryst)

    @classmethod
    def from_cryst(cls, recilat: ReciLattice, is_q0: List[bool], *l_qpts_cryst):
        cryst = []

        for qpts_cryst in l_qpts_cryst:
            cryst.append(qpts_cryst[:3])

        cryst = np.array(cryst)
        # print(cryst.T)

        return cls(recilat, is_q0, cryst)


def sort_cryst_like_BGW(cryst, key_array):
    """Given a cryst array and a primary key, return argsort in BerkeleyGW-compatible style.

    Parameters
    ----------
    cryst :
        Crystal Coordinates of G-vectors
    key_array :
        Key to be used for sorting the cryst vectors.
        Expecting norm squared values of the cryst vectors,
        with q=0 case taken care of as specified in the notes below.

    Returns
    -------
    array of indices to be used to get a sorted array.

    Notes
    -----
    In BerkeleyGW sort style, the primary key is dependent on whether or not q is 0.
    If q is 0 (but the value is slightly shifted, say 0.001), keep crystal ordering for q exactly = 0.

    """

    # Sorting order same as BerkeleyGW
    # Remember that for np.lexsort, the order of keys in the argument
    # is opposite to priority order, so last key is most important.
    # TODO: To be sure, provide facility to read gvecs from epsmat.h5 so that BGW epsmat.h5 can be used with sigma.py etc.
    indices_cryst_sorted = np.lexsort(
        (
            cryst[2, :],
            cryst[1, :],
            cryst[0, :],
            np.around(key_array, 10),
        )
    )

    return indices_cryst_sorted


def reorder_2d_matrix_sorted_gvecs(mat, indices):
    """Given a 2-D matrix and listof indices, reorder rows and columns in order of indices. Convenience function.

    Parameters
    ----------
    mat
        The 2-D matrix
    indices
        List of indices

    Returns
    -------
        ``mat``, with appropriately ordered rows and coumns.
    """
    tiled_indices = np.tile(indices, (len(indices), 1))
    return np.take_along_axis(
        np.take_along_axis(mat, tiled_indices, 1), tiled_indices.T, 0
    )
