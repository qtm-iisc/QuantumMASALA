# %% [markdown]
# ### Tutorial Notebook: G<sub>0</sub>W<sub>0</sub> Approximation 
# 
# In this notebook, we present an example calculation of quasiparticle energies using QuatumMASALA's `gw` module.

# %%
%load_ext autoreload
%autoreload 2

# %%
# Imports
import numpy as np
import sys

sys.path.append(".")

dirname = "./"

# %%
import numpy as np

from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.kpts import KList
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
# from qtm.klist import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger
# qtmconfig.fft_backend = 'mkl_fft'

from mpi4py.MPI import COMM_WORLD
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world)

# Lattice
reallat = RealLattice.from_alat(alat=10.2,  # Bohr
                                a1=[-0.5,  0.0,  0.5],
                                a2=[ 0.0,  0.5,  0.5],
                                a3=[-0.5,  0.5,  0.0])

# Atom Basis
si_oncv = UPFv2Data.from_file('Si_ONCV_PBE-1.2.upf')
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

# mpgrid_shift = (True,True,True)
# kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)
# print(kpts.numkpts)
# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

# -----Spin-polarized (collinear) calculation-----
is_spin, is_noncolin = False, False
# Starting with asymmetric spin distribution else convergence may yield only
# non-magnetized states
mag_start = [0.0]
numbnd = 30  # Ensure adequate # of bands if system is not an insulator

occ = 'fixed'
smear_typ = 'gauss'
e_temp = 0.0001 * RYDBERG

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG

# %%
from qtm.constants import ELECTRONVOLT_HART
from qtm.kpts import KList


kpts = KList(recilat=crystal.recilat, k_coords=kpts.k_cryst, k_weights=np.ones(kpts.k_cryst.shape[1])/kpts.k_cryst.shape[1])

out = scf(dftcomm, crystal, kpts, grho, gwfn,
          numbnd, is_spin, is_noncolin,
          rho_start=mag_start, occ_typ=occ,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status,
          ret_vxc=True)

scf_converged, rho, l_wfn_kgrp, en, vxc = out


print("SCF Routine has exited")
print(qtmlogger)

# print(vxc/ELECTRONVOLT_HART)

# for wfn in l_wfn_kgrp:
#     wfn.evc_gk/=(np.linalg.norm(wfn.evc_gk, axis=-1, keepdims=True))
    # wfn.evl*=2
    # wfn.normalize()

# %%
kpts_q = KList(recilat=crystal.recilat, k_coords=kpts.k_cryst+np.array([[0,0,0.001]]).T, k_weights=np.ones(kpts.k_cryst.shape[1])/kpts.k_cryst.shape[0])

out_q = scf(dftcomm, crystal, kpts_q, grho, gwfn,
          numbnd, is_spin, is_noncolin,
          rho_start=mag_start, occ_typ=occ, smear_typ='gauss', e_temp=e_temp,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status)

scf_converged_q, rho_q, l_wfn_kgrp_q, en_q = out_q


print("Shifted SCF Routine has exited")
print(qtmlogger)

# %% [markdown]
# ### Load Input Files
# Input data is handled by the ``EpsInp`` class.\
# The data can be provided either by constructing the ``EpsInp`` object or by reading BGW-compatible input file ``epsilon.inp``.\
# The attributes have been supplied with docstrings from BerkeleyGW's input specification, so they will be accessible directly in most IDEs.

# %% [markdown]
# ### Load Input Files
# Input data is handled by the ``EpsInp`` class.\
# The data can be provided either by constructing the ``EpsInp`` object or by reading BGW-compatible input file ``epsilon.inp``.\
# The attributes have been supplied with docstrings from BerkeleyGW's input specification, so they will be accessible directly in most IDEs.

# %%
from qtm.gw.io_bgw.epsinp import Epsinp

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
from qtm.gw.io_bgw.sigmainp import Sigmainp
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

# %% [markdown]
# The three main steps involved in the calculation have been mapped to the corresponding functions:
# 1.  ``matrix_elements``: Calculation of Planewave Matrix elements
# 2.  ``polarizability``: Calculation of RPA polarizability matrix $P$
# 3.  ``epsilon_inverse``: Calculation of (static) epsilon-inverse matrix
# 
# <!-- 1.  ``matrix_elements``: Calculation of Planewave Matrix elements
#     $$M_{nn'}({\textbf k},{\textbf q},{\textbf G}) = \bra{n\,{\textbf k}{+}{\textbf q}}e^{i({\textbf q}+{\textbf G})\cdot{\textbf r}}\ket{n'\,\textbf k}$$
#     where the $\textbf G$-vectors included in the calculation satisfy $|\textbf q + \textbf G|^2 < E_{\text{cut}}$.
#     Since this is a convolution in k-space, the time complexity can be reduced from $\mathcal{O}\left(N^2_G\right)$ to $\mathcal{O}\left(N_G\ln N_G\right)$ by using Fast Fourier Transform, where $N_G$  the number of reciprocal lattice vectors in the wavefunction.
#     $$
#     M_{nn'}({\bf k},{\bf q},\{{\bf G}\}) = {\rm FFT}^{-1}\left( \phi^{*}_{n,{\bf k}+{\bf q} }({\bf r}) \phi_{n',{\bf k} }({\bf r}) \right).
#     $$
#     where $\phi_{n',{\bf k}}({\bf r}) = {\rm FFT}\left( \psi_{n\bf k}(\bf G)\right)$. 
#     
# 2.  ``polarizability``: Calculation of RPA polarizability matrix $P$
#     $$
#         P_{\textbf{GG'}}{\left({\textbf q}\;\!;0\right)}=
#         \,\,{}\sum_{n}^{\textrm occ}\sum_{n'}^{\textrm emp}\sum_{{\textbf k}}
#         \frac{
#         \bra{n'\textbf k}e^{-i({\textbf q}+{\textbf G})\cdot{\textbf r}}\ket{n{\textbf k}{+}{\textbf q}}
#         \bra{n{\textbf k}{+}{\textbf q}}e^{i({\textbf q}+{\textbf G'})\cdot{\textbf r}}\ket{n'\textbf k}
#         }{E_{n{\textbf k}{+}{\textbf q}}\,{-}\,E_{n'{\textbf k}}}.
#     $$
# 3.  ``epsilon_inverse``: Calculation of (static) epsilon-inverse matrix
#     $$
#         \epsilon_{\textbf{GG'}}{\left({\textbf q}\;\!\right)}=
#         \delta_{\textbf{GG'}}\,{-}\,v{\left({\textbf q}{+}{\textbf G}\right)} \,
#         P_{\textbf{GG'}}{\left({\textbf q}\;\!\right)}
#     $$
#     where $ v(\textbf{q} + \textbf{G}) = \frac{8\pi}{\left|\textbf q + \textbf G\right|^2} $ is bare Coulomb potential, written in Rydberg units. If this formula is used as-is for the case where $|\textbf q| = |\textbf G| = 0$, the resulting $v\left({\textbf{q=0}, \textbf{G=0}}\;\!\right)$ blows up as $1/q^2$. However, for 3D gapped systems, the matrix elements $\big| M_{nn'}\left({\bf k},{\textbf{q}\to\textbf{0}},{\textbf{G=0}}\right)\big| \sim q$ cancel the Coulomb divergence and $\epsilon_{\textbf{00}}\left({\textbf q\to\textbf{0}}\;\!\right) \sim q^2/q^2$ which is a finite number. In order to calculate $\epsilon_{\textbf{00}}\left({\textbf q=\textbf{0}}\;\!\right)$, we use the scheme specified in BGW2012, wherein $q=0$ is replaced with a small but non-zero value. Since matrix element calculation involves the eigenvectors $\ket{n{\textbf k}{+}{\textbf q}}$, having a non-$\Gamma$-centered $\textbf q\to 0$ point requires mean-field eigenvectors over a shifted $k$-grid. -->

# %%
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
        # if i_q==0:
        #     N=10
        #     print(M.shape)
        #     print(M[0,0,0,:N].real)
        #     break

        # Calculate polarizability matrix (faster, but not memory-efficient)
        chimat = epsilon.polarizability(M)

        # Calculate polarizability matrix (memory-efficient)
        # chimat = epsilon.polarizability_active(i_q)

        # chimat/=2
        
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

# %%
from qtm.gw.sigma import Sigma

outdir = dirname+"temp/"


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

# %%
sigma_ch_cohsex_mat = sigma.sigma_ch_static()    
print("Sigma CH COHSEX")
sigma.pprint_sigma_mat(sigma_ch_cohsex_mat)

# %%
sigma.autosave=False
sigma.print_condition=True
cohsex_result = sigma.calculate_static_cohsex()

# %%
sigma.print_condition=True
sigma_ch_exact_mat = sigma.sigma_ch_static_exact()    
print("Sigma CH COHSEX EXACT")
sigma.pprint_sigma_mat(sigma_ch_exact_mat)

# %%
sigma.print_condition=False
sigma_sx_gpp = sigma.sigma_sx_gpp()    
print("Sigma SX GPP")
sigma.pprint_sigma_mat(sigma_sx_gpp)

# %%
sigma.print_condition=False
sigma_ch_gpp,_ = sigma.sigma_ch_gpp()    
print("Sigma CH GPP")
sigma.pprint_sigma_mat(sigma_ch_gpp)

# %%
gpp_result = sigma.calculate_gpp()


