from quantum_masala import config
from quantum_masala.core import RealLattice, AtomBasis, Crystal
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KPoints

from quantum_masala.core import GSpace, rho_normalize
from quantum_masala.pseudo import rho_generate_atomic
from quantum_masala.dft import scf
from quantum_masala.constants import ELECTRONVOLT_HART, RYDBERG_HART
from time import perf_counter


config.numkgrp = None  # If running in parallel, each kgrp contains 1 process
config.init_pwcomm()
pwcomm = config.pwcomm

start_time = perf_counter()
reallat = RealLattice.from_alat(5.1070, [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5])

fe_oncv = UPFv2Data.from_file('fe', 'Fe_ONCV_PBE-1.2.upf')
fe_atoms = AtomBasis.from_cart('fe', 55.487, fe_oncv, reallat, [0., 0., 0.])

crystal = Crystal(reallat, [fe_atoms])
recilatt = crystal.recilat

kpts = KPoints.mpgrid(crystal, (8, 8, 8), (False, False, False))


ecut_wfn = 20
ecut_rho = 4 * ecut_wfn

grho = GSpace(crystal, ecut_rho, (18, 18, 18))

rhoatomic = rho_generate_atomic(crystal.l_atoms[0], grho)

rho = [0.55, 0.45] * rhoatomic
rho_normalize(rho, 16)
rho.integrate_r()

numbnd = 12

def wfn_init(wfn, ikpt):
    wfn.init_random()

def iter_printer(idxiter, scf_converged, e_err, en):
    if config.pwcomm.world_rank == 0:
        print(f"Iteration # {idxiter+1}, Error: {e_err}")
        print(f"Convergence Status: "
              f"{'NOT' if not scf_converged else ''} Converged")
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
          numbnd=numbnd, noncolin=False, wfn_init=wfn_init,
          xc_params={'exch_name': 'gga_x_pbe', 'corr_name': 'gga_c_pbe'},
          occ='smear', smear_typ='gauss', e_temp=0.005,
          conv_thr=1E-9,
          iter_printer=iter_printer
         )

pwcomm.world_comm.barrier()
if pwcomm.world_rank == 0:
    print("SCF Routine has exited")
