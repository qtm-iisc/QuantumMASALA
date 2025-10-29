import numpy as np
from scipy.special import sph_harm
import math

def real_sph_harm(l, m, theta, phi):
    if m < 0:
        return np.sqrt(2) * np.imag(sph_harm(abs(m), l, phi, theta))
    elif m == 0:
        return sph_harm(0, l, phi, theta).real
    else:
        return np.sqrt(2) * np.real(sph_harm(m, l, phi, theta))


def compute_spherical_harmonics(rl, maxl=3):
    """
    Given an array of points rl, compute the spherical harmonics for l = 0,1,2,3.
    
    The function accepts points in either (3, num_points) or (num_points, 3) format.
    The output array Y has shape (num_points, (maxl+1)**2) with the ordering:
      for each l: m = 0, then for m = 1, -1, 2, -2, ... etc.
      
    Parameters:
      rl : numpy array
           Array of 3D points. Either shape (3, num_points) or (num_points, 3).
      maxl : int
           Maximum l to include (default 3).
           
    Returns:
      Y : ndarray of shape (num_points, (maxl+1)**2)
          The computed real spherical harmonics.
    """
    # Ensure the points are in shape (3, num_points)
    if rl.shape[0] != 3:
        rl = rl.T  # assume original shape was (num_points, 3)
    
    num_points = rl.shape[1]
    Y = np.zeros((num_points, (maxl+1)**2))
    x = rl[0]
    y = rl[1]
    z = rl[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero
    r = np.where(r == 0, 1e-12, r)
    theta = np.arccos(z / r)  # polar angle in [0, pi]
    phi = np.arctan2(y, x)    # azimuthal angle in [-pi, pi]
    
    index = 0
    for l in range(0, maxl+1):
        if l == 0:
            # Only m = 0 exists.
            Y[:, index] = real_sph_harm(0, 0, theta, phi)
            index += 1
        else:
            # m = 0
            Y[:, index] = real_sph_harm(l, 0, theta, phi)
            index += 1
            # For each m = 1,...,l add m and -m in that order.
            for m in range(1, l+1):
                Y[:, index] = real_sph_harm(l, m, theta, phi)
                index += 1
                Y[:, index] = real_sph_harm(l, -m, theta, phi)
                index += 1
    return Y


def rotation_matrix(axis, angle):
    """
    Create a 3x3 rotation matrix for a rotation around a given axis by 'angle' radians.
    
    Parameters:
      axis : array_like
         The axis of rotation.
      angle : float
         The rotation angle in radians.
         
    Returns:
      3x3 numpy array representing the rotation.
    """
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    ux, uy, uz = axis
    return np.array([[cos_a + ux**2*(1-cos_a),      ux*uy*(1-cos_a) - uz*sin_a,  ux*uz*(1-cos_a) + uy*sin_a],
                     [uy*ux*(1-cos_a) + uz*sin_a,    cos_a + uy**2*(1-cos_a),      uy*uz*(1-cos_a) - ux*sin_a],
                     [uz*ux*(1-cos_a) - uy*sin_a,    uz*uy*(1-cos_a) + ux*sin_a,   cos_a + uz**2*(1-cos_a)]])


def compute_d_matrices(sr):
    """
    Compute the d-matrices for the given set of symmetry operations.
    
    Parameters:
      sr : numpy array of shape (nsym, 3, 3)
           Each entry is a 3x3 rotation matrix representing a symmetry operation.
           
    Returns:
      dy1, dy2, dy3 : numpy arrays containing the d-matrices for l=1, l=2, and l=3.
          dy1 : shape (nsym, 3, 3)
          dy2 : shape (nsym, 5, 5)
          dy3 : shape (nsym, 7, 7)
    """
    nsym = sr.shape[0]
    sizes = {1: 3, 2: 5, 3: 7}
    maxm = 7  # number of random points (corresponds to 2*maxl+1)
    
    # Generate 7 random vectors with components in [-0.5, 0.5].
    # These are stored in shape (3, maxm)
    rl = np.random.rand(3, maxm) - 0.5  
    
    # Compute spherical harmonics for the original points.
    # Ylm will have shape (num_points, 16) with num_points == maxm
    ylm = compute_spherical_harmonics(rl)
    
    # Extract blocks corresponding to l=1,2,3.
    # For l=1: Fortran uses indices 1 to 3 (1-based); here we use rows 0:3 and columns 1:4.
    yl1 = ylm[:sizes[1], 1:1+sizes[1]].T  # shape (3, 3)
    yl1_inv = np.linalg.inv(yl1)
    
    # For l=2: use first 5 points and columns 4 to 8.
    yl2 = ylm[:sizes[2], 1+sizes[1]:1+sizes[1]+sizes[2]].T  # shape (5, 5)
    yl2_inv = np.linalg.inv(yl2)
    
    # For l=3: use all 7 points and columns 9 to 16.
    yl3 = ylm[:sizes[3], 1+sizes[1]+sizes[2]:1+sizes[1]+sizes[2]+sizes[3]].T  # shape (7, 7)
    yl3_inv = np.linalg.inv(yl3)
    
    # Allocate arrays for the d-matrices.
    dy1 = np.zeros((nsym, sizes[1], sizes[1]))
    dy2 = np.zeros((nsym, sizes[2], sizes[2]))
    dy3 = np.zeros((nsym, sizes[3], sizes[3]))
    
    eps = 1e-9  # tolerance for orthogonality check
    
    for isym in range(nsym):
        # Rotate the original points.
        srl = sr[isym] @ rl  # srl has shape (3, maxm)
        # Compute spherical harmonics for the rotated points.
        ylms = compute_spherical_harmonics(srl)
        
        # For l=1: extract block (rows 0:3, columns 1:4) and compute d1.
        yl1_rot = ylms[:sizes[1], 1:1+sizes[1]].T  # shape (3, 3)
        d1 = yl1_rot @ yl1_inv
        
        # For l=2: extract block (rows 0:5, columns 4:9) and compute d2.
        yl2_rot = ylms[:sizes[2], 1+sizes[1]:1+sizes[1]+sizes[2]].T  # shape (5, 5)
        d2 = yl2_rot @ yl2_inv
        
        # For l=3: extract block (rows 0:7, columns 9:16) and compute d3.
        yl3_rot = ylms[:sizes[3], 1+sizes[1]+sizes[2]:1+sizes[1]+sizes[2]+sizes[3]].T  # shape (7, 7)
        d3 = yl3_rot @ yl3_inv
        
        # Check orthogonality: d * d^T should be nearly the identity.
        for d, size, label in zip([d1, d2, d3], [sizes[1], sizes[2], sizes[3]], ['l=1', 'l=2', 'l=3']):
            if not np.allclose(np.dot(d, d.T), np.eye(size), atol=eps):
                raise ValueError(f"D_S ({label}) for symmetry operation {isym+1} is not orthogonal")
        # Save the computed d-matrices.
        map = [0,1,-1,2,-2,3,-3]
        dymatrices = [dy1[isym], dy2[isym], dy3[isym]]
        dmatrices = [d1, d2, d3]
        for d, dy in zip(dmatrices, dymatrices):
          s = len(d)
          for m1 in range(s):
            for m2 in range(s):
              m1_, m2_ = map[m1], map[m2]
              dy[m1_, m2_] = d[m1, m2]    
    return dy1, dy2, dy3