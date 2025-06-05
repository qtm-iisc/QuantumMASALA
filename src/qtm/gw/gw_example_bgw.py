from __future__ import annotations
import numpy as np
import sys


sys.path.append(".")

dirname = "../../../tests/bgw/silicon/cohsex/"
# dirname = "../../../tests/bgw/gaas_4/cohsex/"
# dirname = "../gw_old/scripts/results/si_6_nband272/si_6_gw/"

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
epsinp = Epsinp.from_epsilon_inp(filename=dirname + "epsilon.inp")
# print(epsinp)

# There is an analogous system to read SigmaInp
from qtm.interfaces.bgw.sigmainp import Sigmainp

sigmainp = Sigmainp.from_sigma_inp(filename=dirname + "sigma.inp")
# print(sigmainp)

# %% [markdown]
# ### Load WfnData
# Calculation of dielectric matrix requires mean field eigenfunctions. \
# Wavefunction data generated from mean-field codes can be read using the ``wfn2py`` utility, which assumes that the incoming data satisfies BerkeleyGW's [`wfn_h5`](http://manual.berkeleygw.org/3.0/wfn_h5_spec/) specification. The data is stored as a `NamedTuple` object.
#
# For reasons discussed later, we also require wavefunctions on a shifted grid to calculate dielectric matrix at $q\to 0$. This shifted grid dataset will be referred to as `wfnqdata`.
#
# Similarly, the utilities `read_rho` and `read_vxc` can be used to read density and exchange-correlation respectively.

# %%
# wfn2py
from qtm.interfaces.bgw import inp
from qtm.interfaces.bgw.wfn2py import wfn2py

wfndata = wfn2py(
    dirname + "WFN.h5"
)  # , wfn_ecutrho_minus_ecutwfn=epsinp.epsilon_cutoff)
# print(wfndata.__doc__)

wfnqdata = wfn2py(
    dirname + "WFNq.h5"
)  # , wfn_ecutrho_minus_ecutwfn=epsinp.epsilon_cutoff)
# print(wfnqdata.__doc__)

# RHO data
rho = inp.read_rho(dirname + "RHO")

# Vxc data
vxc = inp.read_vxc(dirname + "vxc.dat")

# %% [markdown]
# ### Initialize Epsilon Class
#
# ``Epsilon`` class can be initialized by either directly passing the required `quantummasala.core` objects or by passing the input objects discussed earlier.

# %%
from qtm.gw.core import QPoints
from qtm.gw.epsilon import Epsilon

# Manual initialization
# epsilon = Epsilon(
#     crystal = wfndata.crystal,
#     gspace = wfndata.grho,
#     kpts = wfndata.kpts,
#     kptsq = wfnqdata.kpts,
#     l_wfn = wfndata.l_wfn,
#     l_wfnq = wfnqdata.l_wfn,
#     l_gsp_wfn = wfndata.l_gk,
#     l_gsp_wfnq = wfnqdata.l_gk,
#     qpts = QPoints.from_cryst(wfndata.kpts.recilat, epsinp.is_q0, *epsinp.qpts),
#     epsinp = epsinp,
# )

epsilon = Epsilon.from_data(wfndata=wfndata, wfnqdata=wfnqdata, epsinp=epsinp)

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

        indices_gspace_sorted = sort_cryst_like_BGW(cryst=gkspc.g_cryst, key_array=key)

        # Calculate matrix elements
        M = next(epsilon.matrix_elements(i_q=i_q))

        # Calculate polarizability matrix (faster, but not memory-efficient)
        chimat = epsilon.polarizability(M)

        # Calculate polarizability matrix (memory-efficient)
        # chimat = epsilon.polarizability_active(i_q)

        # Calculate epsilon inverse matrix
        epsinv = epsilon.epsilon_inverse(
            i_q=i_q, polarizability_matrix=chimat, store=True
        )

        epsinv = reorder_2d_matrix_sorted_gvecs(epsinv, indices_gspace_sorted)
        epsilon.l_epsinv[i_q] = epsinv

        # Compare the results with BGW's results
        if i_q == epsilon.qpts.index_q0:
            epsref = epsilon.read_epsmat(dirname + "eps0mat.h5")[0][0, 0]
            if writing:
                epsilon.write_epsmat(
                    filename="test/epsilon/eps0mat_qtm.h5", epsinvmats=[epsinv]
                )
        else:
            epsref = np.array(epsilon.read_epsmat(dirname + "epsmat.h5")[i_q - 1][0, 0])
            epsmats.append(epsinv)

        # Calculate stddev between reference and calculated epsinv matrices
        std_eps = np.std(epsref - epsinv) / np.sqrt(np.prod(list(epsinv.shape)))

        epstol = 1e-16
        if np.abs(std_eps) > epstol:
            print(
                f"Standard deviation exceeded {epstol} tolerance: {std_eps}, for i_q:{i_q}"
            )
            print(epsref[:2, :2])
            print(epsinv[:2, :2])

    if writing:
        epsilon.write_epsmat(filename="test/epsilon/epsmat_qtm.h5", epsinvmats=epsmats)


epsinp.no_min_fftgrid = True
epsilon = Epsilon.from_data(wfndata=wfndata, wfnqdata=wfnqdata, epsinp=epsinp)
calculate_epsilon()

# epsinp.no_min_fftgrid = False
# epsilon = Epsilon.from_data(wfndata=wfndata, wfnqdata=wfnqdata, epsinp=epsinp)
# calculate_epsilon()


# %% [markdown]
# ### Sigma Calculation

# %%
from qtm.gw.sigma import Sigma

outdir = dirname + "temp/"

sigma = Sigma.from_data(
    wfndata=wfndata,
    wfnqdata=wfnqdata,
    sigmainp=sigmainp,
    epsinp=epsinp,
    l_epsmats=epsilon.l_epsinv,
    rho=rho,
    vxc=vxc,
    outdir=outdir,
)

# %%
sigma_sx_cohsex_mat = sigma.sigma_sx_static(yielding=True)
print("Sigma SX COHSEX")
sigma.pprint_sigma_mat(sigma_sx_cohsex_mat)

# %% [markdown]
# from tqdm.auto import tqdm
# sigma_x_mat = sigma.sigma_x(yielding=True)
# print("Sigma X")
# sigma.pprint_sigma_mat(sigma_x_mat)

# %%
sigma_ch_cohsex_mat = sigma.sigma_ch_static()
print("Sigma CH COHSEX")
sigma.pprint_sigma_mat(sigma_ch_cohsex_mat)

# %%
sigma.autosave = False
sigma.print_condition = True
cohsex_result = sigma.calculate_static_cohsex()

# %%
from qtm.interfaces.bgw.sigma_hp_reader import read_sigma_hp

ref_dict = read_sigma_hp(dirname + "sigma_hp.log")
for ik in cohsex_result:
    print("k-point index:", ik)
    qtty = "Eqp1"
    print(np.abs(ref_dict[ik + 1][qtty] - np.around(cohsex_result[ik][qtty], 6)))

# %%
sigma.print_condition = True
sigma_ch_exact_mat = sigma.sigma_ch_static_exact()
print("Sigma CH COHSEX EXACT")
sigma.pprint_sigma_mat(sigma_ch_exact_mat)

# %%
sigma.print_condition = False
sigma_ch_gpp, _ = sigma.sigma_ch_gpp()
print("Sigma CH GPP")
sigma.pprint_sigma_mat(sigma_ch_gpp)

# %%
gpp_result = sigma.calculate_gpp()

# %%
from qtm.interfaces.bgw.sigma_hp_reader import read_sigma_hp

ref_dict = read_sigma_hp(dirname + "../gpp/sigma_hp.log")
for ik in gpp_result:
    print("k-point index:", ik)
    qtty = "Eqp1"
    print(np.abs(ref_dict[ik + 1][qtty] - np.around(gpp_result[ik][qtty], 6)))
