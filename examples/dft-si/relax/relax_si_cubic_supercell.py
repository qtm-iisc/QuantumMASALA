import numpy as np
from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf
from qtm.relax import relax

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger
qtmconfig.fft_backend = 'mkl_fft'

from qtm.pot.force import force, force_ewald, force_local, force_nonloc
from qtm.pot.stress import stress, stress_ewald, stress_local, stress_kinetic, stress_har, stress_nonloc, stress_xc


from mpi4py.MPI import COMM_WORLD
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)
print("is dftcomm.pwgrp_intra None?1st", dftcomm.pwgrp_intra is None, flush=True)

# Lattice
alat=10.2612
reallat = RealLattice.from_alat(
    alat, a1=[1, 0, 0], a2=[0, 1, 0], a3=[0, 0, 1]  # Bohr
)

# Atom Basis
si_oncv = UPFv2Data.from_file('../Si_ONCV_PBE-1.2.upf')
 
si_atoms = BasisAtoms.from_alat(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0,0,0],[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]]),
)

crystal = Crystal(reallat, [si_atoms, ])  # Represents the crystal


#Creating the supercrystal by generating a supercell
supercell_size=[2, 2, 2]
num_cell=supercell_size[0]*supercell_size[1]*supercell_size[2]
super_crystal=crystal.gen_supercell(supercell_size)

print("coordinates before vacancy creation", super_crystal.l_atoms[0].r_alat)

si_atoms=super_crystal.l_atoms[0]
coords_alat_vac=np.delete(si_atoms.r_alat, 32, axis=1)
si_atoms_super=BasisAtoms.from_alat('si', si_oncv, 28.086, super_crystal.reallat, coords_alat_vac.T)


crystal_super=Crystal(super_crystal.reallat, [si_atoms_super, ])

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (1, 1, 1)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal_super, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn= 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal_super.recilat, ecut_rho)
gwfn = grho

#Number of Bands
tot_num = np.sum([sp.numatoms for sp in crystal_super.l_atoms])
extra_band=max(4, int(0.4*tot_num))
numbnd = int(2*tot_num)+extra_band # Ensure adequate # of bands if system is not an insulator
conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-4 * RYDBERG

##Constraint
constraint=np.ones(tot_num)
#constraint[0]=0


occ = 'smear'
smear_typ = 'gauss'
e_temp = 1E-2 * RYDBERG


'''smear_typ=smear_typ,
            e_temp=e_temp,'''

out = relax(dftcomm=dftcomm, 
            constraint=constraint, 
            crystal=crystal_super, 
            kgrid=mpgrid_shape, 
            kshift=mpgrid_shift , 
            ecut_wfn=ecut_wfn,
            numbnd=numbnd, 
            is_spin=False, 
            is_noncolin=False,
            use_symm=True,
            is_time_reversal=True,
            symm_rho=True, 
            rho_start=None, 
            occ_typ='smear',
            smear_typ=smear_typ,
            e_temp=e_temp,
            mix_beta=0.3,
            conv_thr=conv_thr, diago_thr_init=diago_thr_init,
            iter_printer=print_scf_status)

cryst_final, en_final=out

print("cryst_final", [sp.r_alat for sp in cryst_final.l_atoms])




