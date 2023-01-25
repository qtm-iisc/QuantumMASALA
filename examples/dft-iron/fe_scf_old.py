import pyfftw
import numpy as np
from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG
from quantum_masala.core import RealLattice, AtomBasis, Crystal
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KPoints

from quantum_masala.core import GSpace, rho_normalize
from quantum_masala.pseudo import rho_generate_atomic
from quantum_masala.dft import scf
from quantum_masala.constants import ELECTRONVOLT_HART, RYDBERG_HART
from time import perf_counter

config.numkgrp = None
pwcomm = config.pwcomm
config.use_gpu = False

start_time = perf_counter()

# Lattice
reallat = RealLattice.from_alat(alat=5.1070,  # Bohr
                                a1=[ 0.5,  0.5,  0.5],
                                a2=[-0.5,  0.5,  0.5],
                                a3=[-0.5, -0.5,  0.5])

# Atom Basis
fe_oncv = UPFv2Data.from_file('fe', 'Fe_ONCV_PBE-1.2.upf')
fe_atoms = AtomBasis.from_cart('fe', 55.487, fe_oncv, reallat, [0., 0., 0.])

crystal = Crystal(reallat, [fe_atoms,]) # Represents the crystal
recilat = crystal.recilat

print("----------  Crystal Info  ----------")
print(f"Lattice parameter 'alat' : {reallat.alat}  a.u.")
print(f"Unit cell volume         : {reallat.cellvol}  (a.u.)^3")
print(f"Number of atoms/cell     : {sum(typ.numatoms for typ in crystal.l_atoms)}")
print(f"Number of atomic types   : {len(crystal.l_atoms)}")
print(f"Number of electrons      : {crystal.numel}")
print()

print(f"Crystal Axes: coordinates in units of 'alat' ({reallat.alat} a.u.)")
for i, ax in enumerate(reallat.axes_alat):
    print(f"    a({i+1}) = ( {ax[0]:>8.5f}, {ax[0]:>8.5f}, {ax[0]:>8.5f} )")
print()

print(f"Reciprocal Axes: coordinates in units of 'tpiba' ({recilat.tpiba} (a.u.)^-1)")
for i, ax in enumerate(recilat.axes_tpiba):
    print(f"    b({i+1}) = ( {ax[0]:>8.5f}, {ax[0]:>8.5f}, {ax[0]:>8.5f} )")
print()

for i, typ in enumerate(crystal.l_atoms):
    print(f"Atom Species #{i+1}")
    print(f"    Label   : {typ.label}")
    print(f"    Mass    : {typ.mass}")
    print(f"    Valence : {typ.valence}")
    print(f"    Pseudpot: {typ.ppdata.filename}")
    print()
print()

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (8, 8, 8)
mpgrid_shift = (True, True, True)
kpts = KPoints.mpgrid(crystal, mpgrid_shape, mpgrid_shift)

# Alternatively, k-points can be set from input list
# kpts = KPoints.from_tpiba(crystal,
#     [( 0.0000000,   0.0000000,   0.0000000),  0.0019531],
#     [( 0.0000000,  -0.1250000,   0.1250000),  0.0234375],
#     [( 0.0000000,  -0.2500000,   0.2500000),  0.0234375],
#     [( 0.0000000,  -0.3750000,   0.3750000),  0.0234375],
#     [( 0.0000000,   0.5000000,  -0.5000000),  0.0117188],
#     [(-0.1250000,  -0.1250000,   0.2500000),  0.0468750],
#     [(-0.1250000,  -0.2500000,   0.3750000),  0.0937500],
#     [(-0.1250000,   0.6250000,  -0.5000000),  0.0937500],
#     [(-0.2500000,   0.7500000,  -0.5000000),  0.0468750],
#     [(-0.2500000,   0.6250000,  -0.3750000),  0.0468750],
#     [( 0.0000000,   0.0000000,   0.2500000),  0.0117188],
#     [( 0.0000000,  -0.1250000,   0.3750000),  0.0468750],
#     [( 0.0000000,  -0.2500000,   0.5000000),  0.0468750],
#     [( 0.0000000,   0.6250000,  -0.3750000),  0.0234375],
#     [(-0.1250000,  -0.1250000,   0.5000000),  0.0468750],
#     [(-0.1250000,   0.7500000,  -0.3750000),  0.0937500],
#     [(-0.2500000,   0.2500000,   0.2500000),  0.0156250],
#     [(-0.2500000,   0.7500000,  -0.2500000),  0.0156250],
#     [( 0.6250000,  -0.6250000,   0.2500000),  0.0468750],
#     [( 0.5000000,  -0.5000000,   0.2500000),  0.0234375],
#     [( 0.5000000,  -0.6250000,   0.3750000),  0.0468750],
#     [( 0.0000000,   0.0000000,   0.5000000),  0.0117188],
#     [( 0.0000000,  -0.1250000,   0.6250000),  0.0468750],
#     [( 0.0000000,   0.7500000,  -0.2500000),  0.0234375],
#     [(-0.1250000,   0.8750000,  -0.2500000),  0.0468750],
#     [( 0.5000000,  -0.5000000,   0.5000000),  0.0039062],
#     [( 0.0000000,   0.0000000,   0.7500000),  0.0117188],
#     [( 0.0000000,   0.8750000,  -0.1250000),  0.0234375],
#     [( 0.0000000,   0.0000000,  -1.0000000),  0.0019531]
# )

print(f"Number of k-points: {kpts.numk}")
print(f"Cartesian coordinates of k-points in units of tpiba")
for i, (k_cryst, k_weight) in enumerate(kpts):
    k_cart = recilat.cryst2tpiba(k_cryst)
    print(f"    k({i:>3d}) = ( {k_cart[0]:>8.5f}, {k_cart[0]:>8.5f}, {k_cart[0]:>8.5f} )")
print()

# -----Setting up G-Space of calculation-----
ecut_wfn = 40 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal, ecut_rho)
print(f"Kinetic Energy Cutoff (Wavefun): {ecut_wfn / RYDBERG} Ry")
print(f"FFT Grid Shape                 : {grho.grid_shape}")
print(f"Number of G-vectors            : {grho.numg}")


# Initializing starting charge density from superposition of atomic charges
rhoatomic = rho_generate_atomic(crystal.l_atoms[0], grho)

# -----Spin-polarized (collinear) calculation-----
is_spin, is_noncolin = True, False
# Starting with asymmetric spin distribution else convergence may yield only
# non-magnetized states
rho = [0.55, 0.45] * rhoatomic
rho = rho_normalize(rho, 16)
numbnd = 12  # Ensure adequate # of bands if system is not an insulator

occ = 'smear'
smear_typ = 'gauss'
e_temp = 1E-2 * RYDBERG

conv_thr = 1E-10 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG


def iter_printer(idxiter, scf_converged, e_error, diago_thr, diago_avgiter, en):
    if config.pwcomm.world_rank == 0:
        print(f"Iteration # {idxiter+1}, Error: {e_error / RYDBERG} Ry")
        print(f"Convergence Status: "
              f"{'NOT' if not scf_converged else ''} Converged")
        if not scf_converged:
            print(f"Avg Diago Iterations: {diago_avgiter:3.1f}")
            print(f"Diago Threshold     : {diago_thr:.2e} Ry")
        print(f"Run Time: {perf_counter() - start_time}")
        print(f"Total Energy:     {en['total'] / RYDBERG_HART:17.8f} Ry")
        print(f"      one-el:     {en['one_el'] / RYDBERG_HART:17.8f} Ry")
        print(f"     Hartree:     {en['hart'] / RYDBERG_HART:17.8f} Ry")
        print(f"          XC:     {en['xc'] / RYDBERG_HART:17.8f} Ry")
        print(f"       Ewald:     {en['ewald'] / RYDBERG_HART:17.8f} Ry")
        print(f"       Fermi:     {en['fermi'] / ELECTRONVOLT_HART:17.8f} eV")
        print(f"       Smear:     {en['smear'] / RYDBERG_HART:17.8f} Ry")
        print()
    config.pwcomm.world_comm.barrier()


out = scf(crystal=crystal, kpts=kpts, rho_start=rho, symm_rho=True,
          numbnd=numbnd, is_spin=is_spin, is_noncolin=is_noncolin,
          wfn_init=None,
          xc_params={'exch_name': 'gga_x_pbe', 'corr_name': 'gga_c_pbe'},
          occ=occ, smear_typ=smear_typ, e_temp=e_temp,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init, iter_printer=iter_printer)


pwcomm.world_comm.barrier()
if pwcomm.world_rank == 0:
    print("SCF Routine has exited")

print(pw_logger)
