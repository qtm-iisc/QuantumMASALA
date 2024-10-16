from qtm import qtmconfig
qtmconfig.set_gpu(False)
print("GPU enabled:", qtmconfig.gpu_enabled)

# %%
# Imports
import os
import numpy as np
import sys

sys.path.append(".")

dirname = os.path.dirname(__file__)+"/"

# %% [markdown]
# ### DFT Calculation
# 
# We will start with a DFT calculation to get the energy eigenfunctions and eigenvalues.

# %%
import numpy as np

from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.kpts import KList
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger
# qtmconfig.fft_backend = 'mkl_fft'

from qtm.gw.vcoul import Vcoul

Vcoul.N_SAMPLES_COARSE = 1e3
Vcoul.N_SAMPLES = 1e4

from mpi4py.MPI import COMM_WORLD
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world)

# Lattice
reallat = RealLattice.from_alat(alat=10.2,  # Bohr
                                a1=[-0.5,  0.0,  0.5],
                                a2=[ 0.0,  0.5,  0.5],
                                a3=[-0.5,  0.5,  0.0])

# Atom Basis
si_oncv = UPFv2Data.from_file(os.path.join(os.path.dirname(__file__),"Si_ONCV_PBE-1.2.upf"))
si_atoms = BasisAtoms.from_alat('Si', si_oncv, 28.086, reallat,
                               [[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]])

crystal = Crystal(reallat, [si_atoms, ])  # Represents the crystal


# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 4, 4)
x = np.linspace(0,1,mpgrid_shape[0], endpoint=False)
y = np.linspace(0,1,mpgrid_shape[1], endpoint=False)
z = np.linspace(0,1,mpgrid_shape[2], endpoint=False)
xx,yy,zz = np.meshgrid(x,y,z, indexing="ij")
kcryst = np.vstack([xx.flatten(),yy.flatten(),zz.flatten()])

kpts = KList(recilat=crystal.recilat, k_coords=kcryst, k_weights=np.ones(kcryst.shape[1])/kcryst.shape[1])


# -----Setting up G-Space of calculation-----
ecut_wfn = 8 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

# -----Spin-polarized (collinear) calculation-----
is_spin, is_noncolin = False, False
mag_start = [0.0]
numbnd_occ = 4
numbnd_nscf = 10

occ = 'fixed'

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG

# %% [markdown]
# ### DFT: SCF calculation for occupied bands

# %%
from qtm.constants import ELECTRONVOLT_HART
from qtm.kpts import KList


kpts = KList(recilat=crystal.recilat, k_coords=kpts.k_cryst, k_weights=np.ones(kpts.k_cryst.shape[1])/kpts.k_cryst.shape[1])

scf_out = scf(dftcomm, crystal, kpts, grho, gwfn,
          numbnd_occ, is_spin, is_noncolin,
          rho_start=mag_start, occ_typ=occ,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status,
          ret_vxc=True)


print("SCF Routine has exited")
# print(qtmlogger)

# %% [markdown]
# #### DFT: NSCF calculation for unshifted grid
# Observe that `maxiter` has been set to `1` and `diago_thr_init` has been set to a high value.

# %%
rho = scf_out[1].copy()
nscf_out = scf(dftcomm, crystal, kpts, grho, gwfn,
          numbnd_nscf, is_spin, is_noncolin,
          rho_start=rho, 
          occ_typ=occ,
          conv_thr=conv_thr, 
          diago_thr_init=(conv_thr/crystal.numel)/10,
          iter_printer=print_scf_status,
          maxiter=1,
          ret_vxc=True)

scf_converged_nscf, rho_nscf, l_wfn_kgrp, en_nscf, vxc = nscf_out

# %% [markdown]
# #### DFT: NSCF calculation for shifted grid
# 
# Dielectric matrix calculation for the $q\to 0$ point will require energy eigenfunctions for a slightly shifted $k$-grid.

# %%
k_coords_q = kpts.k_cryst+np.array([[0,0,0.001]]).T
k_weights_q = np.ones(k_coords_q.shape[1])/k_coords_q.shape[1]
kpts_q = KList(recilat=crystal.recilat, k_coords=k_coords_q, k_weights=k_weights_q)

rho = scf_out[1].copy()
out_q = scf(dftcomm, crystal, kpts_q, grho, gwfn,
          numbnd_nscf, is_spin, is_noncolin,
          rho_start=rho, 
          occ_typ=occ,
          conv_thr=conv_thr, 
          diago_thr_init=(conv_thr/crystal.numel)/10,
          iter_printer=print_scf_status,
          maxiter=1)

scf_converged_nscf_q, rho_nscf_q, l_wfn_kgrp_q, en_nscf_q = out_q


print("Shifted SCF Routine has exited")
# print(qtmlogger)

# %% [markdown]
# ### Load Input Files
# Input data is handled by the ``EpsInp`` class.\
# The data can be provided either by constructing the ``EpsInp`` object or by reading BGW-compatible input file ``epsilon.inp``.\
# The attributes have been supplied with docstrings from BerkeleyGW's input specification, so they will be accessible directly in most IDEs.

# %%
from qtm.interfaces.bgw.epsinp import Epsinp

# Constructing input manually
# epsinp = Epsinp(epsilon_cutoff=1.2,
#                 use_wfn_hdf5=True,
#                 number_bands=8,
#                 write_vcoul=True,
#                 qpts=[[0.0,0.0,0.0]],
#                 is_q0=[True])

# Reading from epsilon.inp file
epsinp = Epsinp.from_epsilon_inp(filename=dirname+'epsilon.inp')
# print(epsinp)

# There is an analogous system to read SigmaInp
from qtm.interfaces.bgw.sigmainp import Sigmainp
sigmainp = Sigmainp.from_sigma_inp(filename=dirname+'sigma.inp')
# print(sigmainp)

# %% [markdown]
# ### Initialize Epsilon Class
# 
# ``Epsilon`` class can be initialized by either directly passing the required `quantummasala.core` objects or by passing the input objects discussed earlier.

# %%
from qtm.gw.core import QPoints
from qtm.gw.epsilon import Epsilon
from qtm.klist import KList

kpts_gw =   KList(recilat=kpts.recilat,   cryst=kpts.k_cryst.T,   weights=kpts.k_weights)
kpts_gw_q = KList(recilat=kpts_q.recilat, cryst=kpts_q.k_cryst.T, weights=kpts_q.k_weights)

# Manual initialization
epsilon = Epsilon(
    crystal = crystal,
    gspace = grho,
    kpts = kpts_gw,
    kptsq = kpts_gw_q,        
    l_wfn = l_wfn_kgrp,
    l_wfnq = l_wfn_kgrp_q,
    qpts = QPoints.from_cryst(kpts.recilat, epsinp.is_q0, *epsinp.qpts),
    epsinp = epsinp,
)

# epsilon = Epsilon.from_data(wfndata=wfndata, wfnqdata=wfnqdata, epsinp=epsinp)

from tqdm import trange
from qtm.gw.core import reorder_2d_matrix_sorted_gvecs, sort_cryst_like_BGW


def calculate_epsilon(numq=None, writing=False):
    epsmats = []
    if numq is None:
        numq = epsilon.qpts.numq

    for i_q in trange(0, numq, desc="Epsilon> q-pt index"):
        # Create map between BGW's sorting order and QTm's sorting order
        gkspc = epsilon.l_gq[i_q]
        
        if i_q == epsilon.qpts.index_q0:
            key = gkspc.g_norm2
        else:
            key = gkspc.gk_norm2

        indices_gspace_sorted = sort_cryst_like_BGW(
            cryst=gkspc.g_cryst, key_array=key
        )

        # Calculate matrix elements
        M = next(epsilon.matrix_elements(i_q=i_q))

        # Calculate polarizability matrix (faster, but not memory-efficient)
        chimat = epsilon.polarizability(M)

        # Calculate polarizability matrix (memory-efficient)
        # chimat = epsilon.polarizability_active(i_q)
        
        # Calculate epsilon inverse matrix
        epsinv = epsilon.epsilon_inverse(i_q=i_q, polarizability_matrix=chimat, store=True)


        epsinv = reorder_2d_matrix_sorted_gvecs(epsinv, indices_gspace_sorted)
        epsilon.l_epsinv[i_q] = epsinv

        if i_q == epsilon.qpts.index_q0:
            if writing:
                epsilon.write_epsmat(
                    filename="test/epsilon/eps0mat_qtm.h5", epsinvmats=[epsinv]
                )
        else:
            epsmats.append(epsinv)
            
        if False:        
            # Compare the results with BGW's results
            if i_q == epsilon.qpts.index_q0:
                epsref = epsilon.read_epsmat(dirname + "eps0mat.h5")[0][0, 0]
            else:
                epsref = np.array(epsilon.read_epsmat(dirname + "epsmat.h5")[i_q - 1][0, 0])    

            # Calculate stddev between reference and calculated epsinv matrices
            std_eps = np.std(np.abs(epsref) - np.abs(epsinv)) / np.sqrt(np.prod(list(epsinv.shape)))
        
            epstol = 1e-16
            if np.abs(std_eps) > epstol:
                print(f"Standard deviation exceeded {epstol} tolerance: {std_eps}, for i_q:{i_q}")
                print(np.max(np.abs(epsinv-epsref)))
                indices = np.where(np.abs(epsinv)-np.abs(epsref)>1e-5)
                

    if writing:
        epsilon.write_epsmat(filename="test/epsilon/epsmat_qtm.h5", epsinvmats=epsmats)

calculate_epsilon()


# %% [markdown]
# ### Sigma Calculation
# 
# Here we demonstate the calculation of diagonal matrix elements of $\Sigma_{\text{QP}}$. The input parameters for sigma calculation are being read from `sigma.inp` file, but the same parameters can also be provided by manually constructing a `SigmaInp` object. 
# 
# Here we will calculate $\bra{nk}\Sigma_{\text{QP}}\ket{nk}$ for the following k-points:
# - $\Gamma$: `k=(0,0,0)`
# - $L$: `k=(0.5,0.5,0)`
# - $X$: `k=(0,0.5,0)`

# %%
from qtm.gw.sigma import Sigma

outdir = dirname+"temp/"

# Generate data for pytest, for epsilon calculation
print(f"epsinv shapes: {[epsilon.l_epsinv[i].shape for i in range(epsilon.qpts.numq)]}")
print(f"epsinv matrix elements: {epsilon.l_epsinv[0][::25,::40].__repr__()}")
print(f"epsinv matrix elements: {epsilon.l_epsinv[1][::25,::40].__repr__()}")
print(f"epsinv matrix elements: {epsilon.l_epsinv[len(epsilon.l_epsinv)-1][::25,::40].__repr__()}")

sigma = Sigma.from_qtm_scf(
    crystal = crystal,
    gspace = grho,
    kpts = kpts_gw,
    kptsq=kpts_gw_q,
    l_wfn_kgrp=l_wfn_kgrp,
    l_wfn_kgrp_q=l_wfn_kgrp_q,
    sigmainp=sigmainp,
    epsinp = epsinp,
    epsilon=epsilon,
    rho=rho,
    vxc=vxc
)

# Alternatively, the Sigma object can also be intitialized from pw2bgw.x output data (after being procesed by wfn2hdf5.x).
# sigma = Sigma.from_data(
#     wfndata=wfndata,
#     wfnqdata=wfnqdata,
#     sigmainp=sigmainp,
#     epsinp=epsinp,
#     l_epsmats=epsilon.l_epsinv,
#     rho=rho,
#     vxc=vxc,
#     outdir=outdir,
# )

# %%
sigma_sx_cohsex_mat = sigma.sigma_sx_static(yielding=True)    
print("Sigma SX COHSEX")
sigma.pprint_sigma_mat(sigma_sx_cohsex_mat)
print("sigma_sx_cohsex_mat")
print(sigma_sx_cohsex_mat.real.__repr__())

# %%
sigma_ch_cohsex_mat = sigma.sigma_ch_static()    
print("Sigma CH COHSEX")
sigma.pprint_sigma_mat(sigma_ch_cohsex_mat)
# print("sigma_ch_cohsex_mat")
print(sigma_ch_cohsex_mat.real.__repr__())

# %%
# sigma.autosave=False
# sigma.print_condition=True
# cohsex_result = sigma.calculate_static_cohsex()

# %%
sigma.print_condition=True
print("Sigma CH COHSEX EXACT")
sigma_ch_exact_mat = sigma.sigma_ch_static_exact()    
sigma.pprint_sigma_mat(sigma_ch_exact_mat)
# print("sigma_ch_exact_mat")
print(sigma_ch_exact_mat.real.__repr__())

# %%
sigma.print_condition=False
sigma_sx_gpp = sigma.sigma_sx_gpp()    
print("Sigma SX GPP")
sigma.pprint_sigma_mat(sigma_sx_gpp)
# print("sigma_sx_gpp")
print(sigma_sx_gpp.real.__repr__())

# %%
sigma.print_condition=False
sigma_ch_gpp,_ = sigma.sigma_ch_gpp()    
print("Sigma CH GPP")
sigma.pprint_sigma_mat(sigma_ch_gpp)
# print("sigma_ch_gpp")
print(sigma_ch_gpp.real.__repr__())


# %%
# gpp_result = sigma.calculate_gpp()


def test_epsilon():
    eps_0 = np.array([[ 5.95428540e-02-3.37864473e-21j,  1.49924031e-06+5.09204600e-10j],
       [-2.86554509e-02-1.31374333e-03j, -1.32523791e-02-1.71211214e-06j],
       [-5.58481430e+00-3.42986819e-04j, -4.26705422e-03-4.48477910e-07j]])
    eps_1 = np.array([[ 0.19742024+3.67967741e-22j,  0.00092967-5.03016392e-07j],
       [-0.11489718+1.22116267e-04j,  0.00240529+7.38628012e-08j],
       [-0.0564709 +2.60259750e-05j,  0.00290447-4.23463252e-06j]])
    eps_last = np.array([[ 0.19740775+9.37707566e-23j,  0.00092898-2.42755972e-07j],
       [-0.11496078+1.56258408e-05j,  0.00240408-1.34255697e-06j],
       [-0.05648803+6.57957770e-06j, -0.00159294+8.94511240e-07j]])
    
    assert np.allclose(epsilon.l_epsinv[0][::25,::40], eps_0, atol=1e-3)
    assert np.allclose(epsilon.l_epsinv[1][::25,::40], eps_1, atol=1e-3)
    assert np.allclose(epsilon.l_epsinv[len(epsilon.l_epsinv)-1][::25,::40], eps_last, atol=1e-3)

def test_sigma():

    sigma_sx_cohsex_mat_ref = np.array([[11.60475124,  7.86935592,  7.86677731,  7.86800533,  2.89661525,
         2.898401  ,  2.89749192,  2.35361278],
       [10.40527449, 10.40573352,  8.61113031,  8.61053076,  2.51466081,
         2.50957095,  1.30787665,  1.30697261],
       [11.05381796,  9.80507514,  8.23596848,  8.23683262,  2.70010636,
         2.40780632,  2.40555686,  0.74395931]])
    sigma_ch_cohsex_mat_ref = np.array([[-6.47072139, -4.98715241, -4.98686614, -4.98576775, -4.68520464,
        -4.68468544, -4.68473015, -4.57278866],
       [-6.14805008, -6.14769318, -5.46495991, -5.46436191, -4.62661395,
        -4.62631922, -3.46411495, -3.46276922],
       [-6.33710143, -5.93668769, -5.29525874, -5.2959468 , -4.79761171,
        -4.23060002, -4.22974924, -3.02649493]])
    sigma_ch_exact_ref = np.array([[-7.19160213, -7.47961527, -7.47823781, -7.47882685, -7.15841717,
        -7.15774744, -7.15783718, -7.4894029 ],
       [-7.34510177, -7.3452422 , -7.2613748 , -7.26058285, -6.65382874,
        -6.65429553, -7.42099613, -7.41937955],
       [-7.35480779, -7.08732961, -7.41704381, -7.4179515 , -7.12387961,
        -6.968172  , -6.96780801, -6.15291849]])
    
    sigma_sx_gpp_ref = np.array([[11.92733459,  7.98625481,  7.9756274 ,  7.97526351,  3.25914422,
         3.26544561,  3.26587872,  2.98413248],
       [10.55325831, 10.56240978,  8.71244509,  8.70876761,  2.8072632 ,
         2.80281313,  1.70139413,  1.69983925],
       [11.24436541,  9.95840092,  8.34652447,  8.34793943,  3.12364666,
         2.75401925,  2.74753025,  1.08690863]])
    sigma_ch_gpp_ref = np.array([[-5.66895404, -5.24623154, -5.24293021, -5.2405412 , -5.23711835,
        -5.23577388, -5.23711356, -5.44917794],
       [-5.69764999, -5.69859857, -5.42691375, -5.42726713, -4.9193366 ,
        -4.9220091 , -4.59188341, -4.58798738],
       [-5.73242705, -5.50935051, -5.42530025, -5.42645424, -5.3357983 ,
        -4.8049147 , -4.79971894, -3.80045573]])
    
    assert np.allclose(sigma_sx_cohsex_mat.real, sigma_sx_cohsex_mat_ref, atol=5e-3)
    assert np.allclose(sigma_ch_cohsex_mat.real, sigma_ch_cohsex_mat_ref, atol=5e-3)
    assert np.allclose(sigma_ch_exact_mat.real, sigma_ch_exact_ref, atol=5e-3)
    assert np.allclose(sigma_sx_gpp.real, sigma_sx_gpp_ref, atol=5e-3)
    assert np.allclose(sigma_ch_gpp.real, sigma_ch_gpp_ref, atol=5e-3)

test_epsilon()
test_sigma()