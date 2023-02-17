from numpy import array2string
from sys import maxsize, stdout

from quantum_masala.core import Crystal, GSpace, KList
from quantum_masala.constants import RYDBERG, ELECTRONVOLT
from quantum_masala.dft import KSWavefun
from quantum_masala.dft.scf import EnergyData
from quantum_masala import config


def print_worldroot(func):
    pwcomm = config.pwcomm

    def call_func(*args, **kwargs):
        if pwcomm.world_rank == 0:
            func(*args, **kwargs)
        pwcomm.world_comm.barrier()
    return call_func


def print_kgrproot(func):
    pwcomm = config.pwcomm

    stdout.flush()
    def call_func(*args, **kwargs):
        for ikgrp in range(pwcomm.numkgrp):
            if pwcomm.idxkgrp == ikgrp and pwcomm.kgrp_rank == 0:
                func(*args, **kwargs)
            stdout.flush()
            pwcomm.world_comm.barrier()
    return call_func


@print_worldroot
def print_crystal_info(crystal: Crystal):
    reallat, recilat = crystal.reallat, crystal.recilat
    print(f"Lattice parameter 'alat' : {reallat.alat:9.5f}  a.u.")
    print(f"Unit cell volume         : {reallat.cellvol:9.5f}  (a.u.)^3")
    print(f"Number of atoms/cell     : {sum(typ.numatoms for typ in crystal.l_atoms)}")
    print(f"Number of atomic types   : {len(crystal.l_atoms)}")
    print(f"Number of electrons      : {crystal.numel}")
    print()
    print("Crystal Axes: coordinates in units of 'alat' "
          f"({reallat.alat:.5f} a.u.)")
    for i, ai in enumerate(reallat.axes_alat):
        print(f"    a({i+1}) = ({ai[0]:>9.5f}, {ai[1]:>9.5f}, {ai[2]:>9.5f})")
    print()
    print("Reciprocal Axes: coordinates in units of 'tpiba' "
          f"({recilat.tpiba:.5f} (a.u.)^-1)")
    for i, bi in enumerate(recilat.axes_tpiba):
        print(f"    b({i+1}) = ( {bi[0]:>9.5f}, {bi[1]:>9.5f}, {bi[2]:>9.5f})")
    print()
    for i, typ in enumerate(crystal.l_atoms):
        print(f"Atom Species #{i+1}")
        print(f"    Label   : {typ.label}")
        print(f"    Mass    : {typ.mass:5.2f}")
        print(f"    Valence : {typ.valence:5.2f}")
        print(f"    Pseudpot: {typ.ppdata.filename}")
        print(f"               MD5:{typ.ppdata.md5_checksum}")
        print(f"    Coordinates (in units of alat)")
        for j, pos in enumerate(typ.alat.T):
            print(f"        {j+1:3d} - ({pos[0]:>9.5f}, {pos[1]:>9.5f}, {pos[2]:>9.5f})")
        print()
    print()


@print_worldroot
def print_kpoints(kpts: KList):
    recilat = kpts.recilat
    print(f"Number of k-points: {len(kpts)}")
    print(f"Cartesian coordinates of k-points in units of tpiba")
    for i, (k_cryst, k_weight) in enumerate(kpts):
        k_tpiba = recilat.cryst2tpiba(k_cryst)
        print(f"    k({i+1:>3d}) = "
              f"({k_tpiba[0]:>9.5f}, {k_tpiba[1]:>9.5f}, {k_tpiba[2]:>9.5f})"
              f"    weight = {k_weight:9.5f}")
    print()


@print_worldroot
def print_gspc_info(grho: GSpace, gwfn: GSpace):
    print(f"Charge Density (Hard) Grid")
    print(f"Kinetic Energy Cutoff    : {grho.ecut / RYDBERG:.2f} Ry")
    print(f"FFT Grid Shape           : {grho.grid_shape}")
    print(f"Number of G-vectors      : {grho.numg}")
    print()
    print(f"Wavefunction (Smooth) Grid")
    print(f"Kinetic Energy Cutoff    : {(gwfn.ecut / 4) / RYDBERG:.2f} Ry")
    print(f"FFT Grid Shape           : {gwfn.grid_shape}")
    print()


@print_worldroot
def print_scf_status(idxiter: int, scf_runtime: float,
                     scf_converged: bool, e_error: float,
                     diago_thr: float, diago_avgiter: float,
                     en: EnergyData):
    print(f"Iteration # {idxiter + 1}, Run Time: {scf_runtime:5.1f} sec")
    print(f"Convergence Status   : "
          f"{'NOT' if not scf_converged else ''} Converged")
    print(f"SCF Error           : {e_error / RYDBERG:.4e} Ry")
    print(f"Avg Diago Iterations: {diago_avgiter:3.1f}")
    print(f"Diago Threshold     : {diago_thr / RYDBERG:.2e} Ry")
    print()
    print(f"Total Energy:     {en.total / RYDBERG:17.8f} Ry")
    if en.internal is not None:
        print(f"    Internal:     {en.internal / RYDBERG:17.8f} Ry")
    print()
    print(f"      one-el:     {en.one_el / RYDBERG:17.8f} Ry")
    print(f"     Hartree:     {en.hartree / RYDBERG:17.8f} Ry")
    print(f"          XC:     {en.xc / RYDBERG:17.8f} Ry")
    print(f"       Ewald:     {en.ewald / RYDBERG:17.8f} Ry")
    if en.smear is not None:
        print(f"       Smear:     {en.smear / RYDBERG:17.8f} Ry")
    print()
    if en.fermi is not None:
        print(f" Fermi Level:     {en.fermi / ELECTRONVOLT:17.8f} eV")
    else:
        print(f"    HO Level:     {en.HO_level / ELECTRONVOLT:17.8f} eV")
        print(f"    LU Level:     {en.HO_level / ELECTRONVOLT:17.8f} eV")
    print('-'*40)
    print()


@print_kgrproot
def print_bands(l_wfn_kgrp: list[KSWavefun]):
    for wfn in l_wfn_kgrp:
        recilat = wfn.gspc.recilat
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
