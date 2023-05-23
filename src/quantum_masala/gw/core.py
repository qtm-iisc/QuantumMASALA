from typing import List
from quantum_masala.core import ReciprocalLattice, KList
# from quantum_masala.core import FFTGSpace
import numpy as np
from numpy.typing import ArrayLike


class QPoints(KList):
    """ class QPoints: Stores q-grid

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
        recilat: ReciprocalLattice,
        is_q0: List[bool],
        cryst: ArrayLike,
    ):
        """Initialize QPoints

        Parameters
        ----------
        recilat : ReciprocalLattice
            Get from GSpace or KList
        cryst : ArrayLike
            List of q-vectors in crystal coordinates, provided in inp files
        is_q0 : List[bool]
            ``is_q0`` column from inp files
        """
        # cryst_gamma = np.array(cryst)
        #dbg
        print(cryst.shape)

        index_q0 = None
        if is_q0!=None:
            index_q0 = np.where(is_q0)[0][0] 
        else:
            index_q0 = np.argmin(recilat.norm2(cryst.T))
       
        # cryst_gamma[:,index_q0] = 0
        weights = np.ones(cryst.shape[0])
        super().__init__(recilat, cryst, weights)
        self.index_q0 = index_q0
        self.q0vec = cryst[self.index_q0,:]
        # print("q0vec init:", self.q0vec)
        self.numq = cryst.shape[0]
        self.is_q0 = is_q0 #if is_q0!=None else [False]*len(cryst)

    @classmethod
    def from_cryst(cls, recilat: ReciprocalLattice, is_q0: List[bool], *l_qpts_cryst):
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
    cryst : Crystal Coordinates of G-vectors
    key_array : Key to be used for sorting the cryst vectors. 
        Expecting norm squared values of the cryst vectors, 
        with q=0 case taken care of as specified in the notes below.

    Returns
    -------
    array of indices to be used to get a sorted array.

    Notes
    -----
    In BerkeleyGW sort style, the primary key is dependent on whether q is 0.
    If q is 0 (but the value is slightly shifted, say 0.001), keep crystal ordering for q exactly = 0.
    """
    
    # Sorting order same as BerkeleyGW
    # Remember that for np.lexsort, the order of keys in the argument
    # is opposite to priority order, so last key is most important.
    indices_cryst_sorted = np.lexsort(
            (
                cryst[2, :],
                cryst[1, :],
                cryst[0, :],
                np.around(key_array, 5),
            )
        )
    
    return indices_cryst_sorted


# class GSpaceQpt(GSpace):
#     """GSpaceQpt: BerkeleyGW specific GSpace class, one per each q-point
#     Stores sorted G_cryst array, sorted with added qpt offset.

#     Attributes:
#     ----------
#     Same as GSpace, but
#     NOTE:
#     - ``qpt`` stores the corresponding q-point
#     - ``norm2`` contains shifted norms |G+q|^2
#     - ``cryst`` contains G-vectors that were included in |G+q|<=ecut
#     """

#     qpt: Sequence[Tuple[float, float, float]]

#     def __init__(
#         self,
#         crystal: Crystal,
#         ecut: float,
#         grid_shape: Sequence[Tuple[int, int, int]],
#         qpt: Sequence[Tuple[float, float, float]],
#         is_q0:bool = False,
#     ):
#         """Initialise GSpaceQpt 
#         In addition to GSpace attributes, contains g_vectors sorted by |q+G|^2

#         Parameters
#         ----------
#         crystal : Crystal
#         ecut : float
#         grid_shape : Sequence[int, int, int]
#         qpt : Sequence[float,float,float]
#             q-point in crystal coordinates.
#             Will be used as offset while sorting G vectors.
#         reorder: bool
#             WHether to reorder as per KE
#         """

#         # Init super: All validity checks done there
#         super().__init__(crystal, ecut, grid_shape)

#         # Add qpt as an attribute
#         self.qpt = np.array(qpt)

#         # print(self.qpt)

#         # Most of the following is same as corresponding code in GSpace.__init__()
#         # The difference is the following:
#         # - We add qpt to gvecs and then impose cutoff, do sorting etc.
#         # - The sorting is compatible to the one used in BerkeleyGW,
#         #   i.e. Sort by energy, and tie-break using G-vec cooords in order

#         # Generating all points in FFT grid
#         xi = [np.fft.fftfreq(n, 1 / n).astype("i4") for n in self.grid_shape]
#         g_cryst = np.array(np.meshgrid(*xi, indexing="ij")).reshape(3, -1)

#         # Add qpt as shift to g_cryst
#         # if is_q0:
#         #     g_shifted_cryst = g_cryst
#         # else:
#         g_shifted_cryst = g_cryst + self.qpt.T[:, None]

#         # Find |G+q|^2:
#         g_shifted_2 = self.recilat.norm2(g_shifted_cryst)

#         # Put cut-off on |G+q|^2
#         self.gridmask = g_shifted_2 <= self.ecut
#         # NOTE: GSpace defines ecut as |G|^2 < 2*ecut, but BerkeleyGW has |q+G|^2 < Ecut
#         # See the paragraph after eqn.8 in BGW paper: arXiv:1111.4429v3

#         # Indices of G-vecs that made it within cutoff
#         icut = np.nonzero(self.gridmask)[0]
#         if len(icut) < 2:
#             raise ValueError(
#                 f"'ecut' value too small. Only {len(icut)} points within 'ecut'={self.ecut}"
#             )

#         # Usual attribute of GSpace (i.e. super), recalculated
#         self.numg = len(icut)
#         self.cryst = np.array(g_cryst[:, icut], dtype="i4", order="C")

#         self.shifted_norm2 = np.array(g_shifted_2[icut], dtype="f8")

#         # Sorting order same as BerkeleyGW
#         # Remember that for np.lexsort, the order of keys in the argument
#         # is opposite to priority order, so last key is most important.
#         # Primary key is dependent on whether q is 0
#         # if q is 0 (but the value is slightly shifted), keep crystal ordering for q exactly = 0, 
#         # but NOTE: `norm2` will contain shifted grid norms
#         # I know its confusing, blame BGW.
#         i_g_sorted = np.lexsort(
#             (
#                 self.cryst[2, :],
#                 self.cryst[1, :],
#                 self.cryst[0, :],
#                 # np.around(self.shifted_norm2, 5),
#                 np.around(self.recilat.norm2(self.cryst) if is_q0 else self.shifted_norm2, 5),
#             )
#         )

#         # Reorder cryst and norm2 in sorted order
#         self.cryst = self.cryst[:, i_g_sorted]
#         self.norm2 = self.recilat.norm2(self.cryst)
#         self.shifted_norm2 = self.shifted_norm2[i_g_sorted]

#         # !! This needs fixing... Work in progress
#         self.idxgrid = np.unravel_index(icut, self.grid_shape)

#         # Additional quantities required for integrating quantities across a unit cell of bravais lattice
#         self.reallat_cellvol = (2 * np.pi) ** 3 / self.recilat.cellvol
#         self.reallat_dv = self.reallat_cellvol / np.prod(self.grid_shape)
#         self.i_g_sorted = i_g_sorted
#         self.icut = icut

#         # self.fft_dri = FFTGSpace(self)
#         # self.fft_dri=se

#     # def hash_cryst2int(self, gvec):
#     #     return sum([self.grid_shape[i]*gvec[i] for i in range(3)])

#     def cryst_to_norm2(self, l_vecs: Sequence) -> Sequence:
#         """Calculate the norm^2 of a given list of vectors in crystal coordinates.
        
#         Parameters
#         ----------
#         l_vec: Sequence
#             List of vectors in crystal coordinates. shape: (3,:)
        
#         Returns
#         -------
#         np.ndarray of shape (:)
#         """
#         return np.sum(np.square(self.recilat.recvec @ l_vecs), axis=0)
