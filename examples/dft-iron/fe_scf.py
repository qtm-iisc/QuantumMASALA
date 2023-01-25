from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG, ELECTRONVOLT
from quantum_masala.core import RealLattice, AtomBasis, Crystal
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KPoints

from quantum_masala.core import GSpace, rho_normalize
from quantum_masala.pseudo import rho_generate_atomic
from quantum_masala.dft import scf
from time import perf_counter

from numpy import array2string
from sys import maxsize, stdout


config.numkgrp = None
pwcomm = config.pwcomm
print_flag = pwcomm.world_rank == 0
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

if print_flag:
    print("----------  Crystal Info  ----------")
    print(f"Lattice parameter 'alat' : {reallat.alat:9.4f}  a.u.")
    print(f"Unit cell volume         : {reallat.cellvol:9.4f}  (a.u.)^3")
    print(f"Number of atoms/cell     : {sum(typ.numatoms for typ in crystal.l_atoms)}")
    print(f"Number of atomic types   : {len(crystal.l_atoms)}")
    print(f"Number of electrons      : {crystal.numel}")
    print()

    print(f"Crystal Axes: coordinates in units of 'alat' ({reallat.alat:.3f} a.u.)")
    for i, ai in enumerate(reallat.axes_alat):
        print(f"    a({i+1}) = ( {ai[0]:>8.5f}, {ai[1]:>8.5f}, {ai[2]:>8.5f} )")
    print()

    print(f"Reciprocal Axes: coordinates in units of 'tpiba' ({recilat.tpiba:.3f} (a.u.)^-1)")
    for i, bi in enumerate(recilat.axes_tpiba):
        print(f"    b({i+1}) = ( {bi[0]:>8.5f}, {bi[2]:>8.5f}, {bi[2]:>8.5f} )")
    print()

    for i, typ in enumerate(crystal.l_atoms):
        print(f"Atom Species #{i+1}")
        print(f"    Label   : {typ.label}")
        print(f"    Mass    : {typ.mass:5.2f}")
        print(f"    Valence : {typ.valence:5.2f}")
        print(f"    Pseudpot: {typ.ppdata.filename}")
        print()
    print()
pwcomm.world_comm.barrier()

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

if print_flag:
    print(f"Number of k-points: {kpts.numk}")
    print(f"Cartesian coordinates of k-points in units of tpiba")
    for i, (k_cryst, k_weight) in enumerate(kpts):
        k_tpiba = recilat.cryst2tpiba(k_cryst)
        print(f"    k({i+1:>3d}) = "
              f"( {k_tpiba[0]:>8.5f}, {k_tpiba[1]:>8.5f}, {k_tpiba[2]:>8.5f} )")
    print()
pwcomm.world_comm.barrier()

# -----Setting up G-Space of calculation-----
ecut_wfn = 40 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal, ecut_rho)
if print_flag:
    print(f"Kinetic Energy Cutoff (Wavefun): {ecut_wfn / RYDBERG} Ry")
    print(f"FFT Grid Shape                 : {grho.grid_shape}")
    print(f"Number of G-vectors            : {grho.numg}")
    print()
pwcomm.world_comm.barrier()

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
    if print_flag:
        print(f"Iteration # {idxiter+1}, Error: {e_error / RYDBERG} Ry")
        print(f"Convergence Status   : "
              f"{'NOT' if not scf_converged else ''} Converged")
        if not scf_converged:
            print(f"Avg Diago Iterations: {diago_avgiter:3.1f}")
            print(f"Diago Threshold     : {diago_thr / RYDBERG:.2e} Ry")
        print(f"Run Time: {perf_counter() - start_time:5.1f} sec")
        print(f"Total Energy:     {en['total'] / RYDBERG:17.8f} Ry")
        print(f"      one-el:     {en['one_el'] / RYDBERG:17.8f} Ry")
        print(f"     Hartree:     {en['hart'] / RYDBERG:17.8f} Ry")
        print(f"          XC:     {en['xc'] / RYDBERG:17.8f} Ry")
        print(f"       Ewald:     {en['ewald'] / RYDBERG:17.8f} Ry")
        print(f"       Fermi:     {en['fermi'] / ELECTRONVOLT:17.8f} eV")
        print(f"       Smear:     {en['smear'] / RYDBERG:17.8f} Ry")
        print()
    config.pwcomm.world_comm.barrier()


out = scf(crystal=crystal, kpts=kpts, rho_start=rho, symm_rho=True,
          numbnd=numbnd, is_spin=is_spin, is_noncolin=is_noncolin,
          wfn_init=None,
          xc_params={'exch_name': 'gga_x_pbe', 'corr_name': 'gga_c_pbe'},
          occ=occ, smear_typ=smear_typ, e_temp=e_temp,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init, iter_printer=iter_printer)

scf_converged, rho, l_wfn_kgrp, en = out

if print_flag:
    if scf_converged:
        print("SCF Routine has achieved convergence")

stdout.flush()
if scf_converged:
    if print_flag:
        print("Printing KS Band eigenvalues")
    for world_rank in range(pwcomm.world_size):
        if pwcomm.kgrp_rank == 0:
            for wfn in l_wfn_kgrp:
                k_tpiba = recilat.cryst2tpiba(wfn.k_cryst)
                print("            k = "
                      f"( {k_tpiba[0]:>8.5f}, {k_tpiba[1]:>8.5f}, {k_tpiba[2]:>8.5f} )"
                      f"    ({wfn.gkspc.numgk:4d} PWs)")
                for ispin in range(1 + wfn.is_spin):
                    if wfn.is_spin and not wfn.is_noncolin:
                        print(f"    SPIN {'UP' if ispin==0 else 'DOWN'}")
                    print("    Eigenvalues (in eV)")
                    print('', array2string(
                        wfn.evl[ispin] / ELECTRONVOLT, separator="",
                        max_line_width=74, formatter={"float_kind": "{:9.4f}".format},
                        threshold=maxsize)[1:-1], '')
                    print("    Occupation")
                    print('', array2string(
                        wfn.occ[ispin], separator="", max_line_width=74,
                        threshold=maxsize, formatter={"float_kind": "{:9.4f}".format},
                        )[1:-1], '')
                    print()
                print('-'*80)
                stdout.flush()
        pwcomm.world_comm.barrier()

pwcomm.world_comm.barrier()
if print_flag:
    print("SCF Routine has exited")
    print(pw_logger)

