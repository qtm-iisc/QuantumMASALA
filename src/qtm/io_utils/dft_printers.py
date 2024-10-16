from __future__ import annotations
import subprocess
__all__ = ['print_scf_status']
from qtm.dft.kswfn import KSWfn
from qtm.dft.scf import EnergyData
from qtm.constants import RYDBERG, ELECTRONVOLT

def print_eigenvalues(l_kswfn_kgrp: list[list[KSWfn]]):
    print("Printing eigenvalues:")
    for kgrp in l_kswfn_kgrp:
        for kswfn in kgrp:
            print()
            print(f"k-point:    {kswfn.k_cryst}")
            print(f"weight:     {kswfn.k_weight}")
            print(f"Basis size: {kswfn.gkspc.size_g}")
            print(f"  Band  Eigenvalue (eV)  Occupation")
            for i in range(kswfn.numbnd):
                print(f"  {i+1:4d}  {kswfn.evl[i] / ELECTRONVOLT:14.8f}  {kswfn.occ[i]:10.6f}")

def print_scf_status(idxiter: int, scf_runtime: float,
                     scf_converged: bool, e_error: float,
                     diago_thr: float, diago_avgiter: float,
                     en: EnergyData, **kwargs):

    print(f"Iteration # {idxiter + 1}, Run Time: {scf_runtime:5.1f} sec")
    print(f"Convergence Status   : "
          f"{'NOT' if not scf_converged else ''} Converged")
    print(f"SCF Error           : {e_error / RYDBERG:.4e} Ry")
    print(f"Avg Diago Iterations: {diago_avgiter:3.1f}")
    print(f"Diago Threshold     : {diago_thr / RYDBERG:.2e} Ry")
    print()
    print(f"Total Energy:     {en.total / RYDBERG:17.8f} Ry")
    # print(f"Harris-Foulkes Energy:{en.hwf / RYDBERG:17.8f} Ry")
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
        if en.LU_level != None:
            print(f"    LU Level:     {en.LU_level / ELECTRONVOLT:17.8f} eV")
    print('-'*40)
    print()
    
def print_project_git_info():
    try:
        # Command to get the last commit hash, day, date (Date Month, Year), and time (hh:mm:ss) for the project
        command = ['git', 'log', '-1', '--format=%H %ad', '--date=format:%A, %d %B, %Y %H:%M:%S']
        commit_info = subprocess.check_output(command).strip().decode('utf-8')
        commit_hash, commit_datetime = commit_info.split(' ', 1)
        
        print(" - Project Git Info:")
        print(f"    - Commit hash:          {commit_hash}")
        print(f"    - Commit date and time: {commit_datetime}")
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving project git info: {e}")


def print_scf_parameters_old(dftcomm, crystal, grho, gwfn, numbnd, is_spin, is_noncolin, symm_rho, rho_start, wfn_init, libxc_func, occ_typ, smear_typ, e_temp, conv_thr, maxiter, diago_thr_init, iter_printer, mix_beta, mix_dim, dftconfig, ret_vxc, kpts):
    print("Quantum MASALA")
    print_project_git_info()
    print("=========================================")
    print("SCF Parameters:")
    print()
    print(f"- dftcomm:           {dftcomm}")
    print(f"- crystal:           {crystal.__repr__()}")
    print(f"- grho:")
    print(f"    cutoff:          {grho.ecut} Ha")
    print(f"    grid_size:       {grho.grid_shape}")
    print(f"    num_g:           {grho.size_g}")
    print(f"- gwfn:")
    print(f"    cutoff:          {gwfn.ecut} Ha")
    print(f"    grid_size:       {gwfn.grid_shape}")
    print(f"    num_g:           {gwfn.size_g}")
    print(f"- numbnd:            {numbnd}")
    print(f"- is_spin:           {is_spin}")
    print(f"- is_noncolin:       {is_noncolin}")
    print(f"- symm_rho:          {symm_rho}")
    print(f"- rho_start:         {rho_start}")
    print(f"- wfn_init:          {wfn_init}")
    print(f"- libxc_func:        {libxc_func}")
    print(f"- occ_typ:           {occ_typ}")
    print(f"- smear_typ:         {smear_typ}")
    print(f"- e_temp:            {e_temp} Ha")
    print(f"- conv_thr:          {conv_thr} Ha")
    print(f"- maxiter:           {maxiter}")
    print(f"- diago_thr_init:    {diago_thr_init}")
    print(f"- iter_printer:      {iter_printer}")
    print(f"- mix_beta:          {mix_beta}")
    print(f"- mix_dim:           {mix_dim}")
    print(f"- dftconfig:         {dftconfig}")
    print(f"- ret_vxc:           {ret_vxc}")
    print(f"- kpts:")
    print("    kpt[0]  kpt[1]  kpt[2];  weight")
    for row in kpts:
        print(f"    {row[0][0]:7.4f} {row[0][1]:7.4f} {row[0][2]:7.4f}; {row[1]:8.6f}")
    print("\n=========================================")

def print_scf_parameters(dftcomm, crystal, grho, gwfn, numbnd, is_spin, is_noncolin, symm_rho, rho_start, wfn_init, libxc_func, occ_typ, smear_typ, e_temp, conv_thr, maxiter, diago_thr_init, iter_printer, mix_beta, mix_dim, dftconfig, ret_vxc, kpts):
    print("Quantum MASALA")
    print_project_git_info()
    print("=========================================")
    print("SCF Parameters:")
    print()
    print(f"dftcomm         = {dftcomm}")
    print(f"crystal         = {crystal.__repr__()}")
    print(f"grho            = GSpace(crystal.recilat, ecut_rho={grho.ecut}, grid_shape={grho.grid_shape})")
    print(f"grho.num_g      = {grho.size_g}")
    print(f"gwfn            = GSpace(crystal.recilat, ecut_wfn={gwfn.ecut}, grid_shape={gwfn.grid_shape})")
    print(f"gwfn.num_g      = {gwfn.size_g}")
    print(f"numbnd          = {numbnd}")
    print(f"is_spin         = {is_spin}")
    print(f"is_noncolin     = {is_noncolin}")
    print(f"symm_rho        = {symm_rho}")
    print(f"rho_start       = {rho_start}")
    print(f"wfn_init        = {wfn_init}")
    print(f"libxc_func      = {libxc_func}")
    print(f"occ_typ         = {occ_typ}")   
    print(f"smear_typ       = {smear_typ}")
    print(f"e_temp          = {e_temp} # Ha")
    print(f"conv_thr        = {conv_thr} # Ha")
    print(f"maxiter         = {maxiter}")
    print(f"diago_thr_init  = {diago_thr_init}")
    print(f"mix_beta        = {mix_beta}")
    print(f"mix_dim         = {mix_dim}")
    print(f"ret_vxc         = {ret_vxc}")
    print(f"dftconfig       = {dftconfig}")
    print(f"iter_printer    = {iter_printer.__name__}")
    print(f"kpts            =")
    print("    kpt[0]  kpt[1]  kpt[2];  weight")
    for row in kpts:
        print(f"    {row[0][0]:7.4f} {row[0][1]:7.4f} {row[0][2]:7.4f}; {row[1]:8.6f}")
    print("\n=========================================")

def silent_printer(*args, **kwargs):
    pass