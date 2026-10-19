"""Shared real-space lattice-vector generation for Ewald force/stress sums."""
import numpy as np


def transgen(latvec: np.ndarray, rmax: float):
    r"""\textbf{Input}

    r max: the maximum radius we take into account,
            and we keep all the atoms inside that radius.

    latvec: lattice vectors, each column representing a vector.
            Numpy array with dimensions (3,3)

    For example the lattice vectors are given by $\vec{l}_1, \vec{l}_2, \vec{l}_3$\\
    and the maximum radius is $r_{max}$\\\\
    The we make a grid that spans from the point 0,0,0 to approximately $\frac{r_{max}}{|l_1|}$, \frac{r_{max}}{|l_2|}, \frac{r_{max}}{|l_3|}$
    points in x,y,z directions in the first octant of the 3D space. Same is the case for the other of octants. It makes a grid where for
    any point $(i_1, i_2, i_3), |i_j|<r_{max} \forall j=1,2,3$\\\\

    This makes a 3D grid of approximately $2\frac{r_{max}}{|l_1|}$ \times 2\frac{r_{max}}{|l_2|} \times 2\frac{r_{max}}{|l_3|}$ points.

    Now, the lattcie vectors are incorporated into this 3D grid in such a way that each point of this grid contains a vector.
    If the point index is $(i_1,i_2,i_3) \quad  -\frac{r_{max}}{|l_j|} \leq i_j \leq \frac{r_{max}}{|l_j|} \forall j=1,2,3 $,
    then that point contains the vector $\vec{v}= \sum_{j=1}^3 i_j\vec{l}_j$

    After that the 3D grid is flattened for simple handling in the subsequent steps.

    \textbf{Output}:
    A flattened array of dimension: $3 \times (8\prod_{j=1}^3 \frac{r_{max}}{|l_j|})$
    """
    n = np.floor(1 / np.linalg.norm(latvec, axis=1) * rmax).astype('i8') + 2
    ni, nj, nk = n
    l0, l1, l2 = latvec[:, 0], latvec[:, 1], latvec[:, 2]
    i, j, k = np.arange(-ni, ni), np.arange(-nj, nj), np.arange(-nk, nk)
    l0_trans = np.outer(i, l0)[np.newaxis, :, np.newaxis, np.newaxis, :]
    l1_trans = np.outer(j, l1)[np.newaxis, np.newaxis, :, np.newaxis, :]
    l2_trans = np.outer(k, l2)[np.newaxis, np.newaxis, np.newaxis, :, :]
    return np.squeeze(l0_trans + l1_trans + l2_trans).reshape(-1, 3)


def rgen(trans: np.ndarray, dtau: np.ndarray, max_num: float, rmax: float):
    r"""\textbf{Input}:

    trans: An array of the dimension $3 \times number of points in a 3D grid.
    It is basically a flattened 3D grid, where each grid point contains a vector$

    dtau: The interatomic distance by which this grid named "trans" would be shifted.

    max\_num: The maximum number of vectors that will be
    considered for constructing the real-space grid of the Ewald calculation.

    rmax: The maximum distance in $x,y,z$ direction that a real-space grid point can be from the origin.

    \textbf{Description:}
    For context read the documentation of the \texttt{transgen} function.
    In this function, a 3D grid was constructed such for any point $(i_1, i_2, i_3), |i_j|<r_{max} \forall j=1,2,3$

    Now, this grid is shifted by the amount dtau. And it is checked how many of these still satisfy the cristeria
    that every point $(i_1, i_2, i_3), |i_j|<r_{max} \forall j=1,2,3$

    \textbf{Output:}

    r.T: The transpose of the vectors which satisfy the above said criteria.

    r\_norm: The norms of those vectors

    vec\_num: Number of such vectors.
    """
    if rmax == 0:
        raise ValueError("rmax is 0, grid is non-existent.")
    trans_shifted = trans - dtau
    norms = np.linalg.norm(trans_shifted, axis=1)
    mask = (norms < rmax) & (norms ** 2 > 1e-5)
    r = trans_shifted[mask]
    r_norm = norms[mask]
    vec_num = r.shape[0]
    if vec_num >= max_num:
        raise ValueError(f"maximum allowed value of r vectors are {max_num}, got {vec_num}. ")
    return r.T, r_norm, vec_num
