from qtm.constants import RYDBERG_HART


if __name__=="__main__":

    from pprint import pprint
    from qtm.qe.inp.read_inp import PWscfIn
    import sys


    filename = "./P-test.scf.in"
    if len(sys.argv)>1:
        filename = sys.argv[1]
    pwscfin = PWscfIn.from_file(filename)

    pprint(pwscfin)

    from qtm.qe.inp.parse_inp import parse_inp

    pwin, cryst, kpts = parse_inp(pwscfin)

    # print('pwin :',pwin) #debug statement
    # print('cryst :',cryst) #debug statement
    # print('kpts :',kpts) #debug statement

    # ### Run QuantumMASALA with the input

    import numpy as np

    from qtm.constants import RYDBERG, ELECTRONVOLT
    from qtm.lattice import RealLattice
    from qtm.crystal import BasisAtoms, Crystal
    from qtm.pseudo import UPFv2Data
    from qtm.kpts import gen_monkhorst_pack_grid
    from qtm.gspace import GSpace
    from qtm.mpi import QTMComm
    from qtm.dft import DFTCommMod, scf

    from qtm.io_utils.dft_printers import print_scf_status

    from qtm import qtmconfig
    from qtm.logger import qtmlogger
    qtmconfig.fft_backend = 'mkl_fft'

    from mpi4py.MPI import COMM_WORLD
    comm_world = QTMComm(COMM_WORLD)
    dftcomm = DFTCommMod(comm_world, comm_world.size, 1)


    # -----Setting up G-Space of calculation-----
    ecut_wfn = pwin.system.ecutwfc*RYDBERG
    ecut_rho = pwin.system.ecutrho *RYDBERG
    # parse_inp() handles ecutrho=None case appropriately.

    grho = GSpace(cryst.recilat, ecut_rho)
    gwfn = grho

    # -----Spin-polarized (collinear) calculation-----
    is_spin, is_noncolin = (pwin.system.nspin==2), pwin.system.noncolin
    # Starting with asymmetric spin distribution else convergence may yield only
    # non-magnetized states
    mag_start = pwin.system.starting_magnetization
    mag_start = list(mag_start.values())
    print('mag_start :', mag_start) #debug statement
    # exit()
    numbnd = pwin.system.nbnd  # Ensure adequate # of bands if system is not an insulator

    occ = pwin.system.occupations
    if occ == "smearing":
        occ="smear"
    # print(occ)
    smear_typ = pwin.system.smearing
    e_temp = pwin.system.degauss * RYDBERG

    conv_thr = pwin.electrons.conv_thr * RYDBERG
    diago_thr_init = pwin.electrons.diago_thr_init * RYDBERG

    print('diago_thr_init :', diago_thr_init) #debug statement
    print('e_temp :', e_temp) #debug statement
    print('conv_thr :', conv_thr) #debug statement
    print('smear_typ :', smear_typ) #debug statement
    print('is_spin :', is_spin) #debug statement
    print('is_noncolin :', is_noncolin) #debug statement
    print('ecut_wfn :', ecut_wfn) #debug statement
    print('ecut_rho :', ecut_rho) #debug statement

    out = scf(dftcomm, cryst, kpts, grho, gwfn,
            numbnd, 
            is_spin, 
            is_noncolin,
            rho_start=mag_start, 
            occ_typ=occ, 
            smear_typ=smear_typ, 
            e_temp=e_temp,
            conv_thr=conv_thr, 
            diago_thr_init=diago_thr_init,
            iter_printer=print_scf_status)

    scf_converged, rho, l_wfn_kgrp, en = out

    # Print Data for delta benchmark
    if comm_world.rank==0:
        print("     number of atoms/cell      =", sum([atom_type.numatoms for atom_type in cryst.l_atoms]))
        print("     unit-cell volume          =", cryst.reallat.cellvol)
        print("!    total energy              =", en.total/RYDBERG_HART)


    print("SCF Routine has exited")
    print(qtmlogger)



