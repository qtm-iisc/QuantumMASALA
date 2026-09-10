from __future__ import annotations
import datetime
import subprocess
import shutil

__all__ = ["print_scf_status"]
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
                print(
                    f"  {i+1:4d}  {kswfn.evl[i] / ELECTRONVOLT:14.8f}  {kswfn.occ[i]:10.6f}"
                )


def print_scf_status(
    idxiter: int,
    scf_runtime: float,
    scf_converged: bool,
    e_error: float,
    diago_thr: float,
    diago_avgiter: float,
    en: EnergyData,
    **kwargs,
):
    print(f"Iteration # {idxiter + 1}, Run Time: {scf_runtime:5.1f} sec")
    print(f"Convergence Status   : " f"{'NOT' if not scf_converged else ''} Converged")
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
    print("-" * 40)
    print()


def get_git_info():
    """Return git commit information, or None if unavailable."""

    if shutil.which("git") is None:
        return None

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        commit_date = subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                "--format=%ad",
                "--date=rfc",
                # "--date=format:%A, %d %B, %Y %H:%M:%S %Z",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        changed_files = subprocess.check_output(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                "HEAD",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        dirty_files = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-status",
                "HEAD",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        return {
            "hash": commit_hash,
            "date": commit_date,
            "files": changed_files.splitlines() if changed_files else [],
            "dirty_files": dirty_files.splitlines() if dirty_files else [],
        }

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        OSError,
    ):
        return None


def print_project_git_info():
    git_info = get_git_info()

    if git_info is None:
        print("Git information: unavailable")
        return

    print(f"Git information: {git_info['hash']} {git_info['date']}")

    if git_info["dirty_files"]:
        print("Local changes relative to HEAD:")
        for line in git_info["dirty_files"]:
            print(f"  {line}")
    else:
        print("No local changes.")


def print_scf_parameters_old(
    dftcomm,
    crystal,
    grho,
    gwfn,
    numbnd,
    is_spin,
    is_noncolin,
    symm_rho,
    rho_start,
    wfn_init,
    libxc_func,
    occ_typ,
    smear_typ,
    e_temp,
    conv_thr,
    maxiter,
    diago_thr_init,
    iter_printer,
    mix_beta,
    mix_dim,
    dftconfig,
    ret_vxc,
    kpts,
):
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


def print_scf_parameters(
    dftcomm,
    crystal,
    grho,
    gwfn,
    numbnd,
    is_spin,
    is_noncolin,
    symm_rho,
    rho_start,
    wfn_init,
    libxc_func,
    occ_typ,
    smear_typ,
    e_temp,
    conv_thr,
    maxiter,
    diago_thr_init,
    iter_printer,
    mix_beta,
    mix_dim,
    dftconfig,
    ret_vxc,
    kpts,
):
    print("Quantum MASALA")
    print_project_git_info()
    now = datetime.datetime.now()
    print(
        f"Started calculation on {now.strftime('%Y-%m-%d')} at {now.strftime('%H:%M:%S')}."
    )
    print("=========================================")
    print("SCF Parameters:")
    print()
    print(f"dftcomm        = {dftcomm}")
    print(f"crystal        = {crystal.__repr__(indent=14*' ')}")
    print(
        f"grho           = GSpace(crystal.recilat, ecut_rho={grho.ecut}, grid_shape={grho.grid_shape})"
    )
    print(f"grho.num_g     = {grho.size_g}")
    print(
        f"gwfn           = GSpace(crystal.recilat, ecut_wfn={gwfn.ecut}, grid_shape={gwfn.grid_shape})"
    )
    print(f"gwfn.num_g     = {gwfn.size_g}")
    print(f"numbnd         = {numbnd}")
    print(f"is_spin        = {is_spin}")
    print(f"is_noncolin    = {is_noncolin}")
    print(f"symm_rho       = {symm_rho}")
    print(f"rho_start      = {rho_start}")
    print(f"wfn_init       = {wfn_init}")
    print(f"libxc_func     = {libxc_func}")
    print(f"occ_typ        = {occ_typ}")
    print(f"smear_typ      = {smear_typ}")
    print(f"e_temp         = {e_temp} # Ha")
    print(f"conv_thr       = {conv_thr} # Ha")
    print(f"maxiter        = {maxiter}")
    print(f"diago_thr_init = {diago_thr_init}")
    print(f"mix_beta       = {mix_beta}")
    print(f"mix_dim        = {mix_dim}")
    print(f"ret_vxc        = {ret_vxc}")
    print(f"dftconfig      = {dftconfig}")
    print(f"iter_printer   = {iter_printer.__name__}")
    print(f"kpts           =")
    print("    kpt[0]  kpt[1]  kpt[2];  weight")
    for row in kpts:
        print(f"    {row[0][0]:7.4f} {row[0][1]:7.4f} {row[0][2]:7.4f}; {row[1]:8.6f}")
    print("\n=========================================")


def silent_printer(*args, **kwargs):
    pass
