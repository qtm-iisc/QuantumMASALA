__all__ = ["RDist"]
import numpy as np
from qtm.crystal import crystal


def RDist(crystal:crystal,
           coords:np.ndarray,
           rmax:np.ndarray, 
           num_bins:int):
    """
    Calculate the radial distribution function (RDF) for a given crystal structure.

    Parameters:
    crystal: Crystal object containing the lattice vectors and atomic positions.
    coords: Atomic positions in Cartesian coordinates.
    rmax: Maximum distance for RDF calculation.
    num_bins: Number of bins for histogram.

    Returns:
    r: Radial distances.
    g_r: Radial distribution function values.
    """
    
    alat = crystal.reallat.alat
    latvec = np.array(crystal.reallat.axes_alat)
    recvec = np.array(crystal.recilat.axes_tpiba)
    rmax=np.linalg.norm(latvec@rmax)
    rmax*=alat
    
    # Initialize histogram
    hist, bin_edges = np.histogram([], bins=num_bins, range=(0, rmax))

    # Calculate distances between all pairs of atoms
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            dtau = (coords[i] - coords[j]) / alat
            dtau @= recvec.T
            dtau_frac = dtau - np.round(dtau)
            dtau0 = latvec.T @ dtau_frac
            dtau0*=alat
            
            # Calculate distance
            distance = np.linalg.norm(dtau0)
            
            # Update histogram if within range
            if distance < rmax:
                bin_index = int(distance / (rmax / num_bins))
                hist[bin_index] += 1
    
    # Calculate radial distribution function
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    del_r=rmax / num_bins
    volume_element = (4/3) * np.pi * ((bin_centers + del_r/2)**3 - (bin_centers- del_r/2)**3)
    tot_num= coords.shape[0]
    volume=np.linalg.det(latvec)
    volume_element *= (tot_num / volume)


    g_r = hist / volume_element
    
    return bin_centers, g_r