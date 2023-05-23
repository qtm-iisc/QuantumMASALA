# %% [markdown]
# # Epsilon Notebook

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# Imports

# from functools import lru_cache
import numpy as np
from quantum_masala.gw.h5_io.h5_utils import *
import sys


sys.path.append("..")
sys.path.append(".")

# dirname = "./test/bgw/"
dirname = "./scripts/results/si_4_gw/"

# %% [markdown]
# ### Load WFN data

# %%
from quantum_masala.gw.io_bgw import inp

epsinp = inp.read_epsilon_inp(filename=dirname+'epsilon.inp')
sigmainp = inp.read_sigma_inp(filename=dirname+'sigma.inp')

# Epsilon.inp data
# Use __doc__ to print elements
print(epsinp.__doc__)
print(epsinp.options)
print()

# Sigma.inp data
# Use __doc__ to print elements
print(sigmainp.__doc__)
print(sigmainp.options)
print()

# wfn2py
from quantum_masala.gw.io_bgw.wfn2py import wfn2py

wfndata = wfn2py(dirname+'WFN.h5')
print(wfndata.__doc__)

wfnqdata = wfn2py(dirname+'WFNq.h5')
print(wfnqdata.__doc__)

# RHO data
# rho_data = inp.read_rho("./test/bgw/RHO")
# print(rho_data.__doc__)
# print(rho_data.rho)

# Vxc data
# vxc_data = inp.read_vxc('./test/bgw/vxc.dat')
# print(vxc_data.__doc__)
# print("Vxc: vxc values and kpts")
# print(*zip(vxc_data.kpts, vxc_data.vxc), sep="\n\n")



# %%
# Testing FFT
wfn = wfndata.l_wfn[0]
print(wfn.evc_gk.shape)
wfn.gkspc.fft_mod.g2r(wfn.evc_gk[0,1,:]).shape


# %% [markdown]
# ### Initialize Epsilon Class

# %%
from quantum_masala.gw.core import QPoints
from quantum_masala.gw.epsilon import Epsilon

qpts = QPoints.from_cryst(wfndata.kpts.recilat, epsinp.is_q0, *epsinp.qpts)

# epsilon = Epsilon(
#     wfndata.crystal,
#     wfndata.grho,
#     wfndata.kpts,
#     wfnqdata.kpts,
#     wfndata.l_wfn,
#     wfnqdata.l_wfn,
#     wfndata.l_gk,
#     wfnqdata.l_gk,
#     qpts,
#     epsinp,
# )

epsilon = Epsilon.from_data(wfndata=wfndata, wfnqdata=wfnqdata, epsinp=epsinp)
# M = epsilon.Matrix_elements(0)
# # print(M)
# chimat = 4 * epsilon.chi(M)
# epsinv = epsilon.eps_inv(i_q=0, chimat=chimat)

# print(epsinv.shape)
# print(epsinv[0])


# %%
from tqdm import trange
from quantum_masala.gw.core import sort_cryst_like_BGW


def reorder_2d_matrix_sorted_gvecs(a, indices):
    """Given a 2-D matrix and listof indices, reorder rows and columns in order of indices

    Parameters
    ----------
    a
        The 2-D matrix
    indices
        List of indices

    Returns
    -------
        ``a`` with appropriately ordered rows and coumns.
    """
    tiled_indices = np.tile(indices,(len(indices),1))
    return np.take_along_axis(np.take_along_axis(a, tiled_indices, 1), tiled_indices.T, 0)

epsmats = []

for i_q in trange(0, epsilon.qpts.numq, desc="Epsilon> q-pt index"):
    
    # Create map between BGW's sorting order and QTm's sorting order
    gkspc  = epsilon.l_gq[i_q]
    if i_q == epsilon.qpts.index_q0:
        # gk_cryst = gkspc.gspc.cryst[:,gkspc.idxg]
        key = gkspc.g_norm2[gkspc.idxg]
        indices_gspace_sorted = sort_cryst_like_BGW(cryst=gkspc.g_cryst, key_array=key)
    else:
        # gk_cryst = gkspc.cryst
        key = gkspc.norm2
        indices_gspace_sorted = sort_cryst_like_BGW(cryst=gkspc.cryst, key_array=key)
    
    # indices_gspace_sorted = sort_cryst_like_BGW(cryst=gk_cryst, key_array=key)
        

    # Calculate matrix elements
    # M = next(epsilon.matrix_elements(i_q=i_q))
    
    # Calculate polarizability matrix
    # chimat = 4 * epsilon.polarizability(M)

    # Calculate polarizability matrix (memory-efficient)
    chimat = 4 * epsilon.polarizability_active(i_q)

    # Calculate epsilon inverse matrix
    epsinv = epsilon.epsilon_inverse(i_q=i_q, polarizability_matrix=chimat)
    
    # indices = epsilon.l_gq[i_q].gk_indices_tosorted
    epsinv = reorder_2d_matrix_sorted_gvecs(epsinv, indices_gspace_sorted)
    
    # Compare the results with BGW's results
    if i_q == epsilon.qpts.index_q0:
        epsref = epsilon.read_epsmat(dirname + "eps0mat.h5")[0][0, 0]
        indices = epsilon.l_gq[i_q].gk_indices_tosorted
        epsilon.write_epsmat(filename="test/epsilon/eps0mat_qtm.h5", epsinvmats=[epsinv])
    else:
        epsref = np.array(epsilon.read_epsmat(dirname + "epsmat.h5")[i_q - 1][0, 0])
        epsmats.append(epsinv)

    # Calculate stddev between reference and calculated epsinv matrices
    mindim = min(epsref.shape)
    epsref= epsref[:mindim,:mindim]
    std_eps = np.std(epsref-epsinv)/np.sqrt(np.prod(list(epsinv.shape)))
    
    epstol = 1e-14
    if np.abs(std_eps) > epstol:
        print(f"Standard deviation exceeded {epstol} tolerance",std_eps)
        print("i_q",i_q)
        break

epsilon.write_epsmat(filename="test/epsilon/epsmat_qtm.h5", epsinvmats=epsmats)


# %%
# %load_ext line_profiler
# %lprun -f epsilon.matrix_elements epsilon.matrix_elements(i_q=0)

# %prun epsilon.matrix_elements(i_q=0)


# %%
# %load_ext memory_profiler
# %mprun epsilon.matrix_elements(i_q=0)
# %memit epsilon.matrix_elements(i_q=0,yielding=True)

# %% [markdown]
# <!-- ## Issues:
# - Still inaccurate epsilon: Complex part of epsilon? Visual inspection tells that complex part matches exactly to all decimals available (10 digits) -->
# <!-- ##### Fixed, but for later reference:
# - fixed now: Vcoul ordering: Previous code was ordered by decreasing kinetic energy, this one uses Vcoul where ordering has been figured out. -->


