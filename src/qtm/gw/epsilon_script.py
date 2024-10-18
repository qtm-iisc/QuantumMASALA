import datetime
import gc
import os
import numpy as np
from qtm.config import QTMConfig
from qtm.gw.core import QPoints, sort_cryst_like_BGW
from qtm.gw.epsilon import Epsilon
from qtm.interfaces.bgw.wfn2py import wfn2py
import sys
from qtm.interfaces.bgw.epsinp import Epsinp
from qtm.logger import COMM_WORLD

sys.path.append("..")
sys.path.append(".")

# dirname = "./"
# dirname = "./test/bgw/"
dirname = "../gw_old/scripts/results/si_6_nband272/si_6_gw/"
# dirname = "../../../tests/bgw/silicon/cohsex/"

# outdir = "./test/epsilon_large/"
outdir = f"./test/tempdir_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"


verify_epsilon = True

# Load WFN data
if QTMConfig.mpi4py_installed and COMM_WORLD.Get_size() > 1:
    in_parallel = True
else:
    in_parallel = False
print("in_parallel", in_parallel, flush=True)

print_condition = (not in_parallel) or (in_parallel and COMM_WORLD.Get_rank() == 0)


# Epsilon.inp data ____________________________________________________________________
epsinp = Epsinp.from_epsilon_inp(filename=dirname + "epsilon.inp")
# Use __doc__ to print elements
if print_condition:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print("outdir", outdir)
    print(epsinp)
COMM_WORLD.Barrier()
outdir = COMM_WORLD.bcast(outdir, root=0)


# wfn2py _______________________________________________________________________________

wfndata = wfn2py(dirname + "WFN.h5")
if print_condition:
    print("Loaded WFN.h5", wfndata.__doc__, flush=True)

wfnqdata = wfn2py(dirname + "WFNq.h5")
if print_condition:
    print("Loaded WFNq.h5", wfnqdata.__doc__, flush=True)


# Initialize Epsilon Class ______________________________________________________________
qpts = QPoints.from_cryst(wfndata.kpts.recilat, epsinp.is_q0, *epsinp.qpts)


epsinp.no_min_fftgrid = True
epsilon = Epsilon.from_data(wfndata=wfndata, wfnqdata=wfnqdata, epsinp=epsinp)
gc.collect()

from tqdm import trange


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
    tiled_indices = np.tile(indices, (len(indices), 1))
    return np.take_along_axis(
        np.take_along_axis(mat, tiled_indices, 1), tiled_indices.T, 0
    )


epsmats = []
COMM_WORLD.Barrier()
if print_condition:
    print("Data loaded. Starting Epsilon Calculation...")
    print(
        "Epsilon script run started at:",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        flush=True,
    )


def calculate_epsilon(numq=None, writing=False):
    epsmats = []
    if numq is None:
        numq = epsilon.qpts.numq

    if in_parallel:
        proc_rank = COMM_WORLD.Get_rank()
        q_indices = np.arange(epsilon.qpts.numq)
        proc_q_indices = np.array_split(q_indices, COMM_WORLD.Get_size())[proc_rank]
        iterable = proc_q_indices
    else:
        iterable = trange(0, epsilon.qpts.numq, desc="Epsilon> q-pt index")

    # for i_q in trange(0, numq, desc="Epsilon> q-pt index"):
    for i_q in iterable:
        print("i_q", i_q, flush=True)
        # Create map between BGW's sorting order and QTm's sorting order
        gkspc = epsilon.l_gq[i_q]
        if i_q == epsilon.qpts.index_q0:
            key = gkspc.g_norm2
            indices_gspace_sorted = sort_cryst_like_BGW(
                cryst=gkspc.g_cryst, key_array=key
            )
        else:
            key = gkspc.gk_norm2
            indices_gspace_sorted = sort_cryst_like_BGW(
                cryst=gkspc.g_cryst, key_array=key
            )

        # Calculate polarizability matrix (Memory-inefficient, but faster)
        chimat = epsilon.polarizability(next(epsilon.matrix_elements(i_q=i_q)))

        # Calculate polarizability matrix (memory-efficient)
        # chimat = epsilon.polarizability_active(i_q)

        # chimat*=np.prod(epsilon.gspace.grid_shape)**4
        # Calculate epsilon inverse matrix
        epsinv = epsilon.epsilon_inverse(
            i_q=i_q, polarizability_matrix=chimat, store=True
        )

        # indices = epsilon.l_gq[i_q].gk_indices_tosorted
        epsinv = reorder_2d_matrix_sorted_gvecs(epsinv, indices_gspace_sorted)
        epsilon.l_epsinv[i_q] = epsinv

        # Compare the results with BGW's results
        if i_q == epsilon.qpts.index_q0:
            epsref = epsilon.read_epsmat(dirname + "eps0mat.h5")[0][0, 0]
            # indices = epsilon.l_gq[i_q].gk_indices_tosorted
            if writing:
                epsilon.write_epsmat(
                    filename=outdir + "eps0mat_qtm.h5", epsinvmats=[epsinv]
                )
        else:
            epsref = np.array(epsilon.read_epsmat(dirname + "epsmat.h5")[i_q - 1][0, 0])
            epsmats.append(epsinv)

        # Calculate stddev between reference and calculated epsinv matrices
        mindim = min(*epsref.shape)  # , *epsinv.shape)
        epsref = epsref[:mindim, :mindim]
        std_eps = np.std(epsref - epsinv) / np.sqrt(np.prod(list(epsinv.shape)))

        epstol = 1e-15
        if np.abs(std_eps) > epstol:
            print(f"i_q {i_q}: Standard deviation exceeded {epstol} tolerance", std_eps)
            # print("i_q", i_q)
        print(f"i_q {i_q}: Standard deviation: {std_eps}", flush=True)

    COMM_WORLD.Barrier()

    if in_parallel:
        l_epsmats = COMM_WORLD.allgather(epsmats)
        epsmats = []
        for epsmats_proc in l_epsmats:
            epsmats.extend(epsmats_proc)
        COMM_WORLD.Barrier()
        if COMM_WORLD.Get_rank() == 0:
            epsilon.write_epsmat(filename=outdir + "epsmat_qtm.h5", epsinvmats=epsmats)
    else:
        epsilon.write_epsmat(filename=outdir + "epsmat_qtm.h5", epsinvmats=epsmats)

    if print_condition:
        print(
            "Epsilon script run completed at:",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        print(f"Files saved to folder: {outdir}")
        # print(pw_logger)


calculate_epsilon(writing=True)  # numq=1)

COMM_WORLD.Barrier()

# print_condition = (not in_parallel) or (in_parallel and COMM_WORLD.Get_rank()==0)
if in_parallel:
    l_epsmats = COMM_WORLD.allgather(epsmats)
    epsmats = []
    for epsmats_proc in l_epsmats:
        epsmats.extend(epsmats_proc)
    # print(len(epsmats))
    if COMM_WORLD.Get_rank() == 0:
        epsilon.write_epsmat(filename=outdir + "epsmat_qtm.h5", epsinvmats=epsmats)
else:
    epsilon.write_epsmat(filename=outdir + "epsmat_qtm.h5", epsinvmats=epsmats)

if print_condition:
    print(
        "Epsilon script run completed at:",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"Files saved to folder: {outdir}")
