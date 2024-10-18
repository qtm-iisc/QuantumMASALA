import os
import sys

import numpy as np
import pytest
from mpi4py.MPI import COMM_WORLD
from qtm import qtmconfig
from qtm.constants import RYDBERG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import DFTCommMod, EnergyData, scf
from qtm.gspace import GSpace
from qtm.io_utils.dft_printers import print_scf_status
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.lattice import RealLattice
from qtm.logger import qtmlogger
from qtm.mpi import QTMComm
from qtm.pseudo import UPFv2Data

# qtmconfig.fft_backend = "mkl_fft"

comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

# Lattice
reallat = RealLattice.from_alat(
    alat=5.1070, a1=[0.5, 0.5, 0.5], a2=[-0.5, 0.5, 0.5], a3=[-0.5, -0.5, 0.5]  # Bohr
)

# Atom Basis
fe_oncv = UPFv2Data.from_file(
    os.path.join(os.path.dirname(__file__), "Fe_ONCV_PBE-1.2.upf")
)
fe_atoms = BasisAtoms("fe", fe_oncv, 55.487, reallat, np.array([[0.0, 0.0, 0.0]]).T)

crystal = Crystal(
    reallat,
    [
        fe_atoms,
    ],
)  # Represents the crystal


# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 3, 5)
mpgrid_shift = (True, True, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)
print(kpts.numkpts)
# -----Setting up G-Space of calculation-----
ecut_wfn = 15 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

# -----Spin-polarized (collinear) calculation-----
is_spin, is_noncolin = True, False
# Starting with asymmetric spin distribution else convergence may yield only
# non-magnetized states
mag_start = [0.1]
numbnd = 12  # Ensure adequate # of bands if system is not an insulator

occ = "smear"
smear_typ = "gauss"
e_temp = 1e-2 * RYDBERG

conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG


out = scf(
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin,
    is_noncolin,
    rho_start=mag_start,
    occ_typ="smear",
    smear_typ="gauss",
    e_temp=e_temp,
    conv_thr=conv_thr,
    diago_thr_init=diago_thr_init,
    iter_printer=print_scf_status,
)

scf_converged, rho, l_wfn_kgrp, en = out


def test_energy():
    global en
    ref_en = EnergyData(
        total=-108.7572322013999,
        hwf=-108.7572322013999,
        one_el=-19.103027941169813,
        ewald=-91.21243030617246,
        hartree=11.995053402940044,
        xc=-10.436472791329699,
        fermi=1.1520059066614938,
        smear=-0.00035456566796903066,
        internal=-108.75687763573194,
        HO_level=None,
        LU_level=None,
    )
    assert np.isclose(en.total, ref_en.total, atol=1e-5)
    assert np.isclose(en.hwf, ref_en.hwf, atol=1e-5)
    assert np.isclose(en.one_el, ref_en.one_el, atol=1e-5)
    assert np.isclose(en.ewald, ref_en.ewald, atol=1e-8)
    assert np.isclose(en.hartree, ref_en.hartree, atol=1e-5)
    assert np.isclose(en.xc, ref_en.xc, atol=1e-5)


def test_eigenvalues():
    global l_wfn_kgrp
    print("Eigenvalues")
    # print(np.array([wfn[0].evl[1::3] for wfn in l_wfn_kgrp[::7]]).__repr__())
    ref_eigenvalues = np.array(
        [
            [-1.57529711, 0.57049308, 1.05557754, 1.94060137],
            [-1.48899584, 0.78679912, 1.06695803, 1.68511685],
            [-1.65155144, 0.82435361, 1.10766899, 1.52010296],
            [-1.56930836, 0.76526929, 1.13823324, 1.46425011],
            [-1.56130804, 0.77814812, 1.05221834, 1.56026077],
        ], like=l_wfn_kgrp[0][0].evl
    )

    assert np.allclose(
        np.array([wfn[0].evl[1::3] for wfn in l_wfn_kgrp[::7]], like=l_wfn_kgrp[0][0].evl),
        ref_eigenvalues,
        atol=1e-3,
    )


def test_density():
    global rho
    basis_size = 531
    cryst_40 = np.array(
        [
            [0, -2, -1, 2, 1, 3, -1, -2, -3, -5, 1, 1, 1, -2],
            [0, 0, 1, 1, 2, 2, 3, 4, -4, -3, -2, -2, -1, -1],
            [0, 5, 0, -3, 2, -1, 4, 3, 1, 2, 0, -3, 2, -2],
        ], like=l_wfn_kgrp[0][0].evl
    )
    norm2_40 = np.array(
        [
            0.0,
            57.51903849,
            9.08195345,
            27.24586034,
            15.13658908,
            21.19122471,
            39.3551316,
            57.51903849,
            45.40976723,
            57.51903849,
            21.19122471,
            21.19122471,
            33.30049597,
            27.24586034,
        ], like=l_wfn_kgrp[0][0].evl
    )
    data_40 = np.array(
        [
            [
                2.13106827e02 - 1.70798383e-17j,
                -4.24495971e-03 + 9.97537527e-19j,
                1.02073312e01 - 1.17065197e-18j,
                -1.36653492e00 + 5.46288830e-18j,
                -7.41852369e-01 + 5.63136506e-18j,
                -2.26777981e00 + 6.85587460e-18j,
                -3.73153157e-01 - 1.88506238e-18j,
                1.46497267e-03 + 1.17228634e-18j,
                -8.99304945e-02 + 3.75860086e-18j,
                -4.24495971e-03 + 9.97537527e-19j,
                -2.26777981e00 + 6.85587460e-18j,
                -2.26777981e00 + 6.85587460e-18j,
                -1.10652402e00 - 2.36182338e-18j,
                -1.36653492e00 + 5.46288830e-18j,
            ],
            [
                2.02034666e02 + 1.70798383e-17j,
                -4.43759610e-03 + 2.95861336e-18j,
                1.18161765e01 + 6.46300704e-18j,
                -1.89454052e00 - 1.18628421e-18j,
                -1.16823591e00 + 2.88782960e-18j,
                -1.78510624e00 + 5.15961325e-18j,
                -4.07157311e-01 + 1.01703969e-18j,
                1.22450174e-03 + 3.89539635e-18j,
                -1.52288453e-01 + 9.23947342e-18j,
                -4.43759610e-03 + 2.95861336e-18j,
                -1.78510624e00 + 5.15961325e-18j,
                -1.78510624e00 + 5.15961325e-18j,
                -9.60904137e-01 + 2.57076685e-19j,
                -1.89454052e00 - 1.18628421e-18j,
            ],
        ], like=l_wfn_kgrp[0][0].evl
    )
    data_g0 = np.array([213.10682674 - 1.70798383e-17j, 202.03466628 + 1.70798383e-17j], like=l_wfn_kgrp[0][0].evl)

    # print(rho.basis_size)
    # print(rho.gspc.g_cryst[:,::40].__repr__())
    # print(rho.gspc.g_norm2[::40].__repr__())
    # print(rho.data[...,::40].__repr__())
    # print(rho.data_g0.__repr__())

    assert rho.basis_size == basis_size
    assert np.allclose(rho.gspc.g_cryst[:, ::40], cryst_40)
    assert np.allclose(rho.gspc.g_norm2[::40], norm2_40, atol=1e-6)
    assert np.allclose(rho.data[..., ::40], data_40, atol=1e-5)
    assert np.allclose(rho.data_g0, data_g0, atol=1e-5)


test_energy()
test_eigenvalues()
test_density()
