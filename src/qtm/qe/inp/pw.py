from qtm.mpi.gspace import DistGSpace


if __name__=="__main__":

    from pprint import pprint
    from qtm.qe.inp.read_inp import PWscfIn
    import sys

    import numpy as np

    from qtm.constants import RYDBERG, ELECTRONVOLT, RYDBERG_HART
    from qtm.gspace import GSpace
    from qtm.mpi import QTMComm
    from qtm.dft import DFTCommMod, scf

    from qtm.io_utils.dft_printers import print_scf_status

    from qtm.logger import qtmlogger

    # from qtm import qtmconfig
    # qtmconfig.fft_backend = 'mkl_fft'
    # qtmconfig.fft_backend = "pyfftw"

    from mpi4py.MPI import COMM_WORLD
    comm_world = QTMComm(COMM_WORLD)
    

    filename = None
    
    if len(sys.argv)>1:
        filename = sys.argv[1]

    # FIXME: only he root process should fetch the arguments and broadcast them to other processes. 

    def fetch_arg(argstr):
        arg = None
        if argstr in sys.argv:
            index_argstr = sys.argv.index(argstr)
            if len(sys.argv)>index_argstr+1:
                # if sys.argv[index_argstr+1].isnumeric():
                arg=sys.argv[index_argstr+1]
        return arg
    
    def fetch_arg_long_short_default(longname, shortname, default_value, typecasting_func=int):
        if fetch_arg("-"+shortname) != None:
            argval = typecasting_func(fetch_arg("-"+shortname))
        elif fetch_arg("-"+longname) != None:
            argval = typecasting_func(fetch_arg("-"+longname))
        else:
            if comm_world.rank==0:
                print("arg not found:", longname, "; using default value:", default_value)
            argval = default_value
            
        if comm_world.rank==0:
            print(f"{longname}: {argval}")
        return argval
            

    filename = fetch_arg_long_short_default("in", "in", filename, typecasting_func=str)

    npools = fetch_arg_long_short_default("npools", "nk", 1)
    
    ntaskgroups = fetch_arg_long_short_default("ntg", "nt", 1)
    
    nbandgroups = fetch_arg_long_short_default("nband", "nb", 1)

    # print(filename)
    pwscfin = PWscfIn.from_file(filename)
    # if comm_world.rank==0:
    #     pprint(pwscfin)

    from qtm.qe.inp.parse_inp import parse_inp
    pwin, crystal, kpts = parse_inp(pwscfin)
    
        
    # n_pw = comm_world.size
    pwgrp_size = comm_world.size//npools//nbandgroups
    if comm_world.rank==0:
        print("Parallelization info:")
        print(f"npools: {npools}")
        print(f"ntaskgroups: {ntaskgroups}")
        print(f"nbandgroups: {nbandgroups}")
        print(f"pwgrp_size: {pwgrp_size}")
    dftcomm = DFTCommMod(comm_world, npools, pwgrp_size)
    print(dftcomm)
    print()


    # -----Setting up G-Space of calculation-----
    ecut_wfn = pwin.system.ecutwfc * RYDBERG
    ecut_rho = pwin.system.ecutrho * RYDBERG
    # parse_inp() handles ecutrho=None case appropriately.

    grho_serial = GSpace(crystal.recilat, ecut_rho)
    # print(dftcomm.image_comm.size, dftcomm.n_pwgrp)
    if dftcomm.n_pwgrp == dftcomm.image_comm.size:  
        grho = grho_serial
    else:
        grho = DistGSpace(comm_world, grho_serial)
    gwfn = grho

    # -----Spin-polarized (collinear) calculation-----
    is_spin, is_noncolin = (pwin.system.nspin==2), pwin.system.noncolin
    # Starting with asymmetric spin distribution else convergence may yield only
    # non-magnetized states
    mag_start = pwin.system.starting_magnetization
    mag_start = list(mag_start.values())
    numbnd = pwin.system.nbnd  # Ensure adequate # of bands if system is not an insulator
    mix_beta = pwin.electrons.mixing_beta

    occ = pwin.system.occupations
    if occ == "smearing":
        occ="smear"

    smear_typ = pwin.system.smearing
    e_temp = pwin.system.degauss * RYDBERG

    conv_thr = pwin.electrons.conv_thr * RYDBERG
    diago_thr_init = pwin.electrons.diago_thr_init * RYDBERG

    comm_world.barrier()

    out = scf(dftcomm, crystal, kpts, grho, gwfn,
            numbnd, 
            is_spin, 
            is_noncolin,
            rho_start= mag_start,
            occ_typ=occ, 
            smear_typ=smear_typ, 
            mix_beta=mix_beta,
            e_temp=e_temp,
            conv_thr=conv_thr, 
            diago_thr_init=diago_thr_init,
            iter_printer=print_scf_status,
            symm_rho=True)

    scf_converged, rho, l_wfn_kgrp, en = out


    # Print Data for delta benchmark
    if comm_world.rank==0:
        print("     number of atoms/cell      =", sum([atom_type.numatoms for atom_type in crystal.l_atoms]))
        print("     unit-cell volume          =", crystal.reallat.cellvol)
        print("!    total energy              =", en.total/RYDBERG_HART)
        print("!    hwf energy                =", en.hwf/RYDBERG_HART)

        print("SCF Routine has exited")
        if scf_converged:
            print("SCF has converged")
        else:
            print("SCF has NOT converged")

        print(qtmlogger)



