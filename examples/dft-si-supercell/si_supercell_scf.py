from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG, ELECTRONVOLT
from quantum_masala.core import RealLattice, AtomBasis
from quantum_masala.core import Crystal, crystal_gen_supercell
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KPoints

from quantum_masala.core import GSpace
from quantum_masala.pseudo import rho_generate_atomic
from quantum_masala.dft import scf
from time import perf_counter

from numpy import array2string
from sys import maxsize, stdout


config.numkgrp = None  # Only k-point parallelization
pwcomm = config.pwcomm

start_time = perf_counter()

# Lattice
reallat = RealLattice.from_alat(alat=10.2,  # Bohr
                                a1=[-0.5,  0.0,  0.5],
                                a2=[ 0.0,  0.5,  0.5],
                                a3=[-0.5,  0.5,  0.0])

# Atom Basis
si_oncv = UPFv2Data.from_file('Si', 'Si_ONCV_PBE-1.2.upf')
si_atoms = AtomBasis.from_alat('Si', 28.085, si_oncv, reallat,
                               [0., 0., 0.], [0.25, 0.25, 0.25])

crystal = Crystal(reallat, [si_atoms, ])
# Generating supercell
supercell_dim = (4, 4, 4)
print(f"Generating a {supercell_dim} supercell")
crystal = crystal_gen_supercell(crystal, supercell_dim)

reallat = crystal.reallat
recilat = crystal.recilat

kpts = KPoints.gamma(crystal)

ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal, ecut_rho)

rhoatomic = rho_generate_atomic(crystal.l_atoms[0], grho)

is_spin, is_noncolin = False, False
rho_start = rhoatomic
numbnd = crystal.numel // 2

occ = 'fixed'

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG

if pwcomm.world_rank == 0:
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
        print(f"    b({i+1}) = ( {bi[0]:>8.5f}, {bi[1]:>8.5f}, {bi[2]:>8.5f} )")
    print()
    for i, typ in enumerate(crystal.l_atoms):
        print(f"Atom Species #{i+1}")
        print(f"    Label   : {typ.label}")
        print(f"    Mass    : {typ.mass:5.2f}")
        print(f"    Valence : {typ.valence:5.2f}")
        print(f"    Pseudpot: {typ.ppdata.filename}")
        print(f"    Coordinates (in units of alat)")
        for j, pos in enumerate(typ.alat.T):
            print(f"        {j+1:3d} - ( {pos[0]:>7.4f}, {pos[1]:>7.4f}, {pos[2]:>7.4f})")
        print()
    print()
    print(f"Number of k-points: {kpts.numk}")
    print(f"Cartesian coordinates of k-points in units of tpiba")
    for i, (k_cryst, k_weight) in enumerate(kpts):
        k_tpiba = recilat.cryst2tpiba(k_cryst)
        print(f"    k({i+1:>3d}) = "
              f"( {k_tpiba[0]:>8.5f}, {k_tpiba[1]:>8.5f}, {k_tpiba[2]:>8.5f} )"
              f"    weight = {k_weight:8.5f}")
    print()
    print(f"Kinetic Energy Cutoff (Wavefun) : {ecut_wfn / RYDBERG} Ry")
    print(f"Kinetic Energy Cutoff (Ch. Den.): {ecut_wfn / RYDBERG} Ry")
    print(f"FFT Grid Shape                  : {grho.grid_shape}")
    print(f"Number of G-vectors             : {grho.numg}")
    print()
pwcomm.world_comm.barrier()


def iter_printer(idxiter, scf_converged, e_error, diago_thr, diago_avgiter, en):
    if pwcomm.world_rank == 0:
        print(f"Iteration # {idxiter+1}, Error: {e_error / RYDBERG:.4e} Ry")
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
        print(f"    HO Level:     {en['occ_max'] / ELECTRONVOLT:17.8f} eV")
        print()
    config.pwcomm.world_comm.barrier()


out = scf(crystal=crystal, kpts=kpts, rho_start=rho_start, symm_rho=True,
          numbnd=numbnd, is_spin=is_spin, is_noncolin=is_noncolin,
          wfn_init=None,
          xc_params={'exch_name': 'gga_x_pbe', 'corr_name': 'gga_c_pbe'},
          occ=occ, conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=iter_printer
          )

scf_converged, rho, l_wfn_kgrp, en = out

if pwcomm.world_rank == 0:
    print(f"SCF Routine has {'NOT' if not scf_converged else ''} "
          f"achieved convergence")

wfn_gamma = l_wfn_kgrp[0]

stdout.flush()
print_bands = True
if scf_converged and print_bands:
    if pwcomm.world_rank == 0:
        print("Printing KS Band eigenvalues")
        k_tpiba = recilat.cryst2tpiba(wfn_gamma.k_cryst)
        print("            k = "
              f"( {k_tpiba[0]:>8.5f}, {k_tpiba[1]:>8.5f}, {k_tpiba[2]:>8.5f} )"
              f"    ({wfn_gamma.gkspc.numgk:4d} PWs)")
        for ispin in range(1 + wfn_gamma.is_spin):
            if wfn_gamma.is_spin and not wfn_gamma.is_noncolin:
                print(f"    SPIN {'UP' if ispin==0 else 'DOWN'}")
            print("    Eigenvalues (in eV)")
            print('', array2string(
                wfn_gamma.evl[ispin] / ELECTRONVOLT, separator="",
                max_line_width=74, formatter={"float_kind": "{:9.4f}".format},
                threshold=maxsize)[1:-1], '')
            print("    Occupation")
            print('', array2string(
                wfn_gamma.occ[ispin], separator="", max_line_width=74,
                threshold=maxsize, formatter={"float_kind": "{:9.4f}".format},
                )[1:-1], '')
            print()
            print('-'*80)
pwcomm.world_comm.barrier()


if pwcomm.world_rank == 0:
    print("SCF Routine has exited")
    print(pw_logger)
