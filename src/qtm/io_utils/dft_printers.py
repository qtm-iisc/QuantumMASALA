from __future__ import annotations
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
        print(f"    LU Level:     {en.LU_level / ELECTRONVOLT:17.8f} eV")
    print('-'*40)
    print()