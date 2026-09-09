"""Simple, DFT-free unit tests for `qtm.pot.ewald.compute`: the classical
Ewald real+reciprocal-space electrostatic energy of the point-charge ions in
a crystal.

Rather than pinning the result to a specific literature "Madelung constant"
(risky to get right from memory, given unit/nearest-neighbor-distance/
double-counting convention differences across sources), this checks
convention-independent physical invariants that any correct implementation
must satisfy: extensivity under a supercell redescription of the same
physical lattice, translational invariance, invariance under reordering
species, and the trivial zero-charge limit.

Uses synthetic species built from bare valence-electron-count ints (as in
'../crystal_tests/test_crystal.py') -- no pseudopotential, no SCF.
"""
import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.crystal import BasisAtoms, Crystal
from qtm.gspace import GSpace
from qtm.lattice import RealLattice
from qtm.pot import ewald

reallat = RealLattice.from_alat(alat=8.0, a1=[1.0, 0, 0], a2=[0, 1.0, 0], a3=[0, 0, 1.0])
r1 = np.array([[0.1], [0.2], [0.15]])
r2 = np.array([[0.6], [0.55], [0.7]])
ecut = 30.0


def _crystal(r1_cryst=r1, r2_cryst=r2, q1=1, q2=-1):
    sp1 = BasisAtoms("P", q1, None, reallat, r1_cryst)
    sp2 = BasisAtoms("N", q2, None, reallat, r2_cryst)
    return Crystal(reallat, [sp1, sp2])


crystal = _crystal()
gspc = GSpace(crystal.recilat, ecut)


def test_extensive_under_supercell_redescription():
    # The same physical infinite lattice, described as a (2,1,1) supercell
    # (twice the atoms, twice the cell volume), must have exactly twice the
    # Ewald energy per cell -- this is a fundamental, convention-independent
    # consistency requirement of the real+reciprocal-space split, not tied
    # to any particular structure's known Madelung constant.
    e0 = ewald.compute(crystal, gspc)
    sup = crystal.gen_supercell((2, 1, 1))
    gspc_sup = GSpace(sup.recilat, ecut)
    e_sup = ewald.compute(sup, gspc_sup)
    assert np.isclose(e_sup, 2 * e0, rtol=1e-8)


def test_invariant_under_rigid_translation_of_the_basis():
    e0 = ewald.compute(crystal, gspc)
    shift = 0.37
    shifted = _crystal((r1 + shift) % 1.0, (r2 + shift) % 1.0)
    e_shifted = ewald.compute(shifted, gspc)
    assert np.isclose(e_shifted, e0, atol=1e-10)


def test_invariant_under_species_reordering():
    e0 = ewald.compute(crystal, gspc)
    sp1 = BasisAtoms("P", 1, None, reallat, r1)
    sp2 = BasisAtoms("N", -1, None, reallat, r2)
    swapped = Crystal(reallat, [sp2, sp1])
    e_swapped = ewald.compute(swapped, gspc)
    assert np.isclose(e_swapped, e0, atol=1e-10)


def test_zero_charge_gives_zero_energy():
    zero_crystal = _crystal(q1=0, q2=0)
    e_zero = ewald.compute(zero_crystal, gspc)
    assert np.isclose(e_zero, 0.0, atol=1e-12)
