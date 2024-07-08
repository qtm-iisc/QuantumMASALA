from __future__ import annotations
from datetime import datetime
import subprocess
__all__ = ['print_scf_status']
from qtm.dft.scf import EnergyData
from qtm.constants import RYDBERG, ELECTRONVOLT


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


def print_scf_parameters(dftcomm, crystal, grho, gwfn, numbnd, is_spin, is_noncolin, symm_rho, rho_start, wfn_init, libxc_func, occ_typ, smear_typ, e_temp, conv_thr, maxiter, diago_thr_init, iter_printer, mix_beta, mix_dim, dftconfig, ret_vxc, kpts):
    print(f"Quantum MASALA SCF calculation started on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    print_project_git_info()
    print("=====================================================================")
    print("SCF Parameters:")
    print()
    print(f"\tdftcomm: {dftcomm}")
    print(f"\tcrystal: {crystal}")
    print(f"\tgrho:")
    print(f"\t\tcutoff: {grho.ecut}")
    print(f"\t\tgrid_size: {grho.grid_shape}")
    print(f"\t\tnum_g: {grho.size_g}")
    print(f"\tgwfn:")
    print(f"\t\tcutoff: {gwfn.ecut}")
    print(f"\t\tgrid_size: {gwfn.grid_shape}")
    print(f"\t\tnum_g: {gwfn.size_g}")
    print(f"\tnumbnd: {numbnd}")
    print(f"\tis_spin: {is_spin}")
    print(f"\tis_noncolin: {is_noncolin}")
    print(f"\tsymm_rho: {symm_rho}")
    print(f"\trho_start: {rho_start}")
    print(f"\twfn_init: {wfn_init}")
    print(f"\tlibxc_func: {libxc_func}")
    print(f"\tocc_typ: {occ_typ}")
    print(f"\tsmear_typ: {smear_typ}")
    print(f"\te_temp: {e_temp}")
    print(f"\tconv_thr: {conv_thr}")
    print(f"\tmaxiter: {maxiter}")
    print(f"\tdiago_thr_init: {diago_thr_init}")
    print(f"\titer_printer: {iter_printer}")
    print(f"\tmix_beta: {mix_beta}")
    print(f"\tmix_dim: {mix_dim}")
    print(f"\tdftconfig: {dftconfig}")
    print(f"\tret_vxc: {ret_vxc}")
    print(f"\tkpts:")
    print("\t\tkpt[0]  kpt[1]  kpt[2];  weight")
    for row in kpts:
        print(f"\t\t{row[0][0]:7.4f} {row[0][1]:7.4f} {row[0][2]:7.4f}; {row[1]:8.6f}")
    print("=====================================================================")