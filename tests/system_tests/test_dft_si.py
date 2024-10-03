import os
import sys
import pytest

import numpy as np
from qtm.constants import RYDBERG
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf, EnergyData

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger

# qtmconfig.fft_backend = "mkl_fft"

from mpi4py.MPI import COMM_WORLD

comm_world = QTMComm(COMM_WORLD)

# Only k-pt parallelization:
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)


# Lattice
reallat = RealLattice.from_alat(
    alat=10.2, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]  # Bohr
)

# Atom Basis
si_oncv = UPFv2Data.from_file(os.path.join(os.path.dirname(__file__),"Si_ONCV_PBE-1.2.upf"))
si_atoms = BasisAtoms(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]).T,
)

crystal = Crystal(reallat, [si_atoms])  # Represents the crystal


# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 3, 1)
mpgrid_shift = (False,True, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 5 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

numbnd = crystal.numel // 2  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

out = scf(
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin=False,
    is_noncolin=False,
    symm_rho=True,
    rho_start=None,
    occ_typ="fixed",
    conv_thr=conv_thr,
    diago_thr_init=diago_thr_init,
    iter_printer=print_scf_status,
)

scf_converged, rho, l_wfn_kgrp, en = out    

    
def test_energy():
    global en
    ref_en = EnergyData(total=-7.640125014954195, hwf=-7.640125014954195, one_el=2.6943002677360086, ewald=-8.449879284940081, hartree=0.5345757262790113, xc=-2.419121724029133, fermi=None, smear=None, internal=None, HO_level=0.2258873352546717, LU_level=None)
    assert np.isclose(en.total, ref_en.total, atol=1e-5)
    assert np.isclose(en.hwf, ref_en.hwf, atol=1e-5)
    assert np.isclose(en.one_el, ref_en.one_el, atol=1e-5)
    assert np.isclose(en.ewald, ref_en.ewald, atol=1e-8)
    assert np.isclose(en.hartree, ref_en.hartree, atol=1e-5)
    assert np.isclose(en.xc, ref_en.xc, atol=1e-5)
    assert np.isclose(en.HO_level, ref_en.HO_level, atol=1e-5)

    
def test_eigenvalues():
    global l_wfn_kgrp
    ref_eigenvalues = np.array([[-0.17022504,  0.16252761,  0.22588733,  0.22588734],
       [-0.15295112,  0.14666559,  0.18508893,  0.19158568],
       [-0.08615121,  0.01177918,  0.14573473,  0.17641629],
       [-0.12734933,  0.07916298,  0.13864487,  0.20814514],
       [-0.09497278, -0.00494601,  0.19909315,  0.19909315],
       [-0.07740766,  0.02630399,  0.11573031,  0.16016657],
       [-0.02246724, -0.02246723,  0.13688391,  0.13688391]])
    
    assert np.allclose(np.array([wfn[0].evl for wfn in l_wfn_kgrp]), ref_eigenvalues, atol=1e-4)
    

def test_density():
    global rho
    basis_size = 411
    cryst_40 = np.array([[ 0,  0,  1,  3, -1,  4, -2,  0, -1,  3, -3],
        [ 0,  0,  1,  2,  2,  3, -5, -3, -2, -1, -1],
        [ 0, -3,  2,  0, -2,  4, -2, -4,  1,  0, -2]])
    norm2_40 = np.array([ 0.        , 10.24526408,  3.0356338 , 10.24526408, 13.28089789,
        16.31653169, 19.35216549, 19.35216549,  7.58908451, 13.66035211,
            7.58908451])
    data_40 = np.array([[ 5.21066558e+001+3.04453322e-050j,
            1.44181634e-001-3.33137527e-019j,
            1.42408416e+000+1.74072741e-016j,
            1.37952406e-001+1.23412818e-019j,
            1.98880687e-003+4.71562066e-019j,
            7.59122737e-003+1.94214448e-018j,
            -7.80327799e-006-1.51460401e-018j,
            2.68389559e-004-1.10400848e-018j,
            -3.78236396e-100+3.78236396e-100j,
            -1.55613810e-002-1.95310687e-018j,
            3.78236396e-100-3.78236396e-100j]])
    data_g0 = np.array([52.10665581+3.04453322e-50j])
    
    assert rho.basis_size == basis_size
    assert np.allclose(rho.gspc.g_cryst[:,::40], cryst_40)
    assert np.allclose(rho.gspc.g_norm2[::40], norm2_40, atol=1e-6)
    assert np.allclose(rho.data[...,::40], data_40, atol=1e-5)
    assert np.allclose(rho.data_g0, data_g0, atol=1e-5)
    
test_energy()
test_eigenvalues()
test_density()