# %% [markdown]
# # Epsilon Notebook

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# Imports

# from functools import lru_cache
import datetime
import gc
import os
import numpy as np
from quantum_masala import pw_logger
from quantum_masala.core.pwcomm import _MPI4PY_INSTALLED, COMM_WORLD
from quantum_masala.gw.h5_io.h5_utils import *
import sys

from quantum_masala.gw.io_bgw.epsinp import Epsinp
from quantum_masala.gw.io_bgw.sigmainp import Sigmainp


sys.path.append("..")
sys.path.append(".")

dirname = "./"
# dirname = "./test/bgw/"
# dirname = "./scripts/results/si_4_gw_cohsex_nn25000/"
# dirname = "./scripts/results/si_4_10_ryd_printing/cohsex/"
# dirname = "./scripts/results/si_6_nband272/si_6_gw/"
# dirname = "/home/agrimsharma/codes/QuantumMASALA/src/quantum_masala/gw/scripts/results/si_6_nband272_pristine_cohsex/si_6_gw/"
# dirname = "/home/agrimsharma/codes/QuantumMASALA/src/quantum_masala/gw/scripts/results/si_4_10_ryd_printing/cohsex/"
# dirname = "/home/agrimsharma/codes/QuantumMASALA/src/quantum_masala/gw/scripts/results/si_4_20_ryd_pristine/cohsex/"
# dirname = "/home/agrimsharma/codes/QuantumMASALA/src/quantum_masala/gw/scripts/results/si_4_nband20_10_ryd_printing/cohsex/"

# outdir = "./test/epsilon_large/"
outdir = f"./test/tempdir_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/" 


verify_epsilon = False

# Load WFN data
from quantum_masala.gw.io_bgw import inp

if _MPI4PY_INSTALLED and COMM_WORLD.Get_size() > 1:
    in_parallel=True
else:
    in_parallel = False
print("in_parallel", in_parallel, flush=True)

print_condition = (not in_parallel) or (in_parallel and COMM_WORLD.Get_rank()==0)


# Epsilon.inp data
# epsinp = inp.read_epsilon_inp(filename=dirname+'epsilon.inp')
epsinp = Epsinp.from_epsilon_inp(filename=dirname+'epsilon.inp')
# Use __doc__ to print elements
if print_condition:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print("outdir", outdir)
    print(epsinp)
COMM_WORLD.Barrier()
outdir = COMM_WORLD.bcast(outdir, root=0)


# Sigma.inp data
sigmainp = Sigmainp.from_sigma_inp(filename=dirname+'sigma.inp')
# sigmainp = inp.read_sigma_inp(filename=dirname+'sigma.inp')
# Use __doc__ to print elements
if print_condition:
    print(sigmainp)

COMM_WORLD.Barrier()

# wfn2py
from quantum_masala.gw.io_bgw.wfn2py import wfn2py

wfndata = wfn2py(dirname+'WFN.h5')
if print_condition:
    print("Loaded WFN.h5",wfndata.__doc__, flush=True)

wfnqdata = wfn2py(dirname+'WFNq.h5')
if print_condition:
    print("Loaded WFNq.h5",wfnqdata.__doc__, flush=True)

# exit()
# RHO data
# rho_data = inp.read_rho("./test/bgw/RHO")
# print(rho_data.__doc__)
# print(rho_data.rho)

# Vxc data
# vxc_data = inp.read_vxc('./test/bgw/vxc.dat')
# print(vxc_data.__doc__)
# print("Vxc: vxc values and kpts")
# print(*zip(vxc_data.kpts, vxc_data.vxc), sep="\n\n")


# Initialize Epsilon Class
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
gc.collect()
# M = epsilon.Matrix_elements(0)
# # print(M)
# chimat = 4 * epsilon.chi(M)
# epsinv = epsilon.eps_inv(i_q=0, chimat=chimat)

# print(epsinv.shape)
# print(epsinv[0])

from tqdm import trange
from quantum_masala.gw.core import sort_cryst_like_BGW


def reorder_2d_matrix_sorted_gvecs(mat, indices):
    """Given a 2-D matrix and listof indices, reorder rows and columns in order of indices

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
    tiled_indices = np.tile(indices,(len(indices),1))
    return np.take_along_axis(np.take_along_axis(mat, tiled_indices, 1), tiled_indices.T, 0)

epsmats = []
COMM_WORLD.Barrier()
if print_condition:
    print("Data loaded. Starting Epsilon Calculation...")
    print(
        "Epsilon script run started at:",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        flush=True
    )


if in_parallel:
    proc_rank = COMM_WORLD.Get_rank()
    q_indices = np.arange(epsilon.qpts.numq)
    proc_q_indices =  np.array_split(q_indices, COMM_WORLD.Get_size())[proc_rank]
    iterable = proc_q_indices
else:
    iterable = trange(0, epsilon.qpts.numq, desc="Epsilon> q-pt index")

for i_q in iterable:    
    print(f"Rank: {COMM_WORLD.Get_rank()}, i_q:{i_q} started", flush=True)
    # Create map between BGW's sorting order and QTm's sorting order
    # DEBUG: Original statement was the first branch of `if`
    # FIXME: Revert back when done
    if True:
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
    else:
        gkspc  = epsilon.l_gq[i_q]
        indices_gspace_sorted = gkspc.gk_indices_fromsorted

    # Calculate matrix elements, followed by polarizability matrix
    chimat = 4 * epsilon.polarizability(next(epsilon.matrix_elements(i_q=i_q)))

    # Calculate polarizability matrix (memory-efficient)
    # chimat = 4 * epsilon.polarizability_active(i_q)

    # Calculate epsilon inverse matrix
    epsinv = epsilon.epsilon_inverse(i_q=i_q, polarizability_matrix=chimat)
    
    # indices = epsilon.l_gq[i_q].gk_indices_tosorted
    epsinv = reorder_2d_matrix_sorted_gvecs(epsinv, indices_gspace_sorted)
    
    # Compare the results with BGW's results
    if i_q == epsilon.qpts.index_q0:
        if verify_epsilon:
            epsref = epsilon.read_epsmat(dirname + "eps0mat.h5")[0][0, 0]
        indices = epsilon.l_gq[i_q].gk_indices_tosorted
        epsilon.write_epsmat(filename=outdir+"eps0mat_qtm.h5", epsinvmats=[epsinv])
    else:
        if verify_epsilon:
            epsref = np.array(epsilon.read_epsmat(dirname + "epsmat.h5")[i_q - 1][0, 0])
        # if not in_parallel:
        epsmats.append(epsinv)
    epsilon.write_epsmat(filename=outdir+f"epsmat_{i_q}_qtm.h5", epsinvmats=[epsinv])

    # Calculate stddev between reference and calculated epsinv matrices
    if verify_epsilon:
        mindim = min(epsref.shape)
        epsref= epsref[:mindim,:mindim]
        std_eps = np.std(epsref-epsinv)/np.sqrt(np.prod(list(epsinv.shape)))
        
        epstol = 1e-14
        if np.abs(std_eps) > epstol:
            print(f"Standard deviation exceeded {epstol} tolerance",std_eps)
            print("i_q",i_q)
            # break

    print(f"Rank: {COMM_WORLD.Get_rank()}, i_q:{i_q} done", flush=True)

COMM_WORLD.Barrier()

# print_condition = (not in_parallel) or (in_parallel and COMM_WORLD.Get_rank()==0)
if in_parallel:
    l_epsmats = COMM_WORLD.allgather(epsmats)
    epsmats = []
    for epsmats_proc in l_epsmats:
        epsmats.extend(epsmats_proc)
    # print(len(epsmats))
    if COMM_WORLD.Get_rank()==0:
        epsilon.write_epsmat(filename=outdir+"epsmat_qtm.h5", epsinvmats=epsmats)
else:
    epsilon.write_epsmat(filename=outdir+"epsmat_qtm.h5", epsinvmats=epsmats)

if print_condition:

    print(
        "Epsilon script run completed at:",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"Files saved to folder: {outdir}")
    print(pw_logger)