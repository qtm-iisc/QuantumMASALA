import os
import numpy as np

from quantum_masala import config, pw_logger
from quantum_masala.constants import RYDBERG
from quantum_masala.core import RealLattice, AtomBasis, Crystal, GField
from quantum_masala.pseudo import UPFv2Data
from quantum_masala.core import KList

from quantum_masala.core import GSpace

from quantum_masala.utils.dft_printers import (
    print_crystal_info, print_kpoints, print_gspc_info,
)

from quantum_masala.tddft_gamma.wfn_bpar import WavefunBgrp
from quantum_masala.tddft_gamma.optical import dipole_response

config.numkgrp = 1
pwcomm = config.pwcomm

# Lattice
reallat = RealLattice.from_alat(alat=32.0,
                                a1=[1., 0., 0.],
                                a2=[0., 1., 0.],
                                a3=[0., 0., 0.83])

# Atom Basis
c_oncv = UPFv2Data.from_file('C', 'C_ONCV_PBE-1.2.upf')
h_oncv = UPFv2Data.from_file('H', 'H_ONCV_PBE-1.2.upf')

c_atoms = AtomBasis.from_angstrom(
    'C', 12.011, c_oncv, reallat,
    (5.633200899, 6.320861303, 5.000000000),
    (6.847051545, 8.422621957, 5.000000000),
    (8.060751351, 7.721904557, 5.000000000),
    (8.060707879, 6.320636665, 5.000000000),
    (6.846898786, 5.620067381, 5.000000000),
    (5.633279551, 7.722134449, 5.000000000),
)

h_atoms = AtomBasis.from_angstrom(
    'H', 1.008, h_oncv, reallat,
    (6.847254360, 9.512254789, 5.000000000),
    (9.004364510, 8.266639340, 5.000000000),
    (9.004297495, 5.775895755, 5.000000000),
    (6.846845929, 4.530522778, 5.000000000),
    (4.689556006, 5.776237709, 5.000000000),
    (4.689791688, 8.267023318, 5.000000000),
)

crystal = Crystal(reallat, [c_atoms, h_atoms])
kpts = KList.gamma(crystal)

ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
gspc_rho = GSpace(crystal, ecut_rho)
gspc_wfn = gspc_rho

print_crystal_info(crystal)
print_kpoints(kpts)
print_gspc_info(gspc_rho, gspc_wfn)

is_spin, is_noncolin = False, False
numbnd = crystal.numel // 2

wfn_gamma = WavefunBgrp(gspc_wfn, (0.0, 0.0, 0.0), 1.0,
                        numbnd, is_spin, is_noncolin)

wfn_gamma_dat = np.load('wfn.npz')
l_ibnd = wfn_gamma.l_ibnd
wfn_gamma.evc_gk[:] = wfn_gamma_dat['evc_gk'][:, l_ibnd]
wfn_gamma.occ[:] = wfn_gamma_dat['occ'][:, l_ibnd]

rho_dat = np.load('rho.npy')
rho_scf = GField.from_array(gspc_rho, rho_dat)

gamma = 1E-4
time_step = 0.05
numsteps = 10000

dip_z = dipole_response(crystal, rho_scf, wfn_gamma,
                        time_step, numsteps, gamma, 'z')


fname = 'dipz.npy'
if os.path.exists(fname) and os.path.isfile(fname):
    os.remove(fname)
np.save(fname, dip_z)
