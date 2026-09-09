"""Simple, DFT-free unit tests for `qtm.crystal`: `Crystal.numel`,
`Crystal.gen_supercell`'s volume/atom-count/valence bookkeeping,
`CrystalSymm.check_supercell`'s filtering of spurious pure-translation
"symmetries", and `CrystalSymm.filter_frac_trans`'s FFT-grid-commensurability
criterion.

Uses synthetic species built with a bare valence-electron-count int (a
documented `BasisAtoms` use case for "routines where pseudopotentials are not
involved") instead of a real pseudopotential file, and a plain simple-cubic
lattice -- no SCF, no UPF file needed. `spglib` (already a dependency) is
still exercised via `Crystal`'s own symmetry detection.
"""
import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.crystal import BasisAtoms, Crystal
from qtm.lattice import RealLattice

reallat = RealLattice.from_alat(alat=5.0, a1=[1.0, 0, 0], a2=[0, 1.0, 0], a3=[0, 0, 1.0])


# ----- Crystal.numel ----------------------------------------------------------
def test_numel_sums_valence_times_numatoms_across_species():
    sp1 = BasisAtoms("A", 4, None, reallat, np.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]]))
    sp2 = BasisAtoms("B", 6, None, reallat, np.array([[0.25], [0.25], [0.25]]))
    crystal = Crystal(reallat, [sp1, sp2])
    assert crystal.numel == 4 * 2 + 6 * 1


# ----- Crystal.gen_supercell ---------------------------------------------------
def test_gen_supercell_scales_volume_atom_count_and_preserves_valence():
    atoms = BasisAtoms("X", 4, None, reallat, np.array([[0.1], [0.2], [0.3]]))
    crystal = Crystal(reallat, [atoms])
    repeats = (2, 1, 3)

    sup = crystal.gen_supercell(repeats)
    assert np.isclose(sup.reallat.cellvol, np.prod(repeats) * crystal.reallat.cellvol)
    assert sup.l_atoms[0].numatoms == np.prod(repeats) * atoms.numatoms

    # Regression test: gen_supercell used to pass 'sp.ppdata' (None, for a
    # species built from a bare valence int) straight through, which
    # BasisAtoms.__init__ interprets as valence=-1 -- silently corrupting
    # the electron count instead of preserving it.
    assert sup.l_atoms[0].valence == atoms.valence
    assert sup.numel == np.prod(repeats) * crystal.numel


def test_gen_supercell_atoms_are_images_of_the_original_under_lattice_translations():
    atoms = BasisAtoms("X", 4, None, reallat, np.array([[0.1], [0.2], [0.3]]))
    crystal = Crystal(reallat, [atoms])
    repeats = (2, 2, 1)
    sup = crystal.gen_supercell(repeats)

    expected_cart = set()
    for n1 in range(repeats[0]):
        for n2 in range(repeats[1]):
            for n3 in range(repeats[2]):
                offset = crystal.reallat.cryst2cart(np.array([n1, n2, n3], dtype="f8"))
                cart = crystal.reallat.cryst2cart(atoms.r_cryst[:, 0]) + offset
                expected_cart.add(tuple(np.round(cart, decimals=8)))

    actual_cart = {
        tuple(np.round(sup.l_atoms[0].r_cart[:, i], decimals=8))
        for i in range(sup.l_atoms[0].numatoms)
    }
    assert expected_cart == actual_cart


# ----- CrystalSymm.check_supercell ---------------------------------------------
def test_check_supercell_drops_spurious_pure_translations():
    # A (2,1,1) supercell of a simple-cubic 1-atom basis is, taken at face
    # value, a tetragonal cell with 2 atoms related by a pure half-lattice-
    # vector translation. spglib (correctly) reports that translation as a
    # "symmetry" of the given cell+basis; 'check_supercell' exists
    # specifically to drop such spurious identity-rotation operations that
    # only exist because the cell isn't given in its primitive form.
    atoms = BasisAtoms("X", 4, None, reallat, np.array([[0.0], [0.0], [0.0]]))
    crystal = Crystal(reallat, [atoms])
    sup = crystal.gen_supercell((2, 1, 1))

    assert sup.symm.numsymm == 16  # tetragonal point group (4/mmm), filtered

    from qtm.crystal.crystal import CrystalSymm

    try:
        CrystalSymm.check_supercell = False
        sup_nocheck = Crystal(sup.reallat, sup.l_atoms)
        assert sup_nocheck.symm.numsymm == 32  # includes the spurious translation

        rot = sup_nocheck.symm.reallat_rot
        trans = sup_nocheck.symm.reallat_trans
        is_pure_translation = np.all(
            rot == np.eye(3, dtype="i4"), axis=(1, 2)
        ) & (np.linalg.norm(trans, axis=1) > 1e-8)
        assert np.any(is_pure_translation)
    finally:
        CrystalSymm.check_supercell = True


# ----- CrystalSymm.filter_frac_trans --------------------------------------------
def test_filter_frac_trans_keeps_only_grid_commensurate_translations():
    atoms = BasisAtoms("X", 4, None, reallat, np.array([[0.0], [0.0], [0.0]]))
    crystal = Crystal(reallat, [atoms])

    ident = np.eye(3, dtype="i4")
    dtype = crystal.symm.symm.dtype
    synthetic = np.array(
        [
            (ident, [0.0, 0.0, 0.0], ident),  # commensurate (identity)
            (ident, [0.25, 0.0, 0.0], ident),  # commensurate: 0.25*4 = 1
            (ident, [0.3, 0.0, 0.0], ident),  # NOT commensurate: 0.3*4 = 1.2
            (ident, [0.5, 0.5, 0.5], ident),  # commensurate: *4 = (2,2,2)
        ],
        dtype=dtype,
    )
    crystal.symm.symm = synthetic

    crystal.symm.filter_frac_trans((4, 4, 4))
    assert crystal.symm.numsymm == 3
    assert np.allclose(
        crystal.symm.reallat_trans, [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )


def test_filter_frac_trans_is_idempotent():
    atoms = BasisAtoms("X", 4, None, reallat, np.array([[0.0], [0.0], [0.0]]))
    crystal = Crystal(reallat, [atoms])
    crystal.symm.filter_frac_trans((4, 4, 4))
    first = crystal.symm.symm.copy()
    crystal.symm.filter_frac_trans((4, 4, 4))
    assert np.array_equal(crystal.symm.symm, first)
