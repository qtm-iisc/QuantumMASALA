from __future__ import annotations

__all__ = ["Crystal", "CrystalSymm"]

import numpy as np
from spglib import get_symmetry

from qtm.config import qtmconfig
from qtm.lattice import RealLattice, ReciLattice
from qtm.crystal.basis_atoms import BasisAtoms

from qtm.msg_format import *


class Crystal:
    """Represents the structure of a Crystal in QuantumMASALA

    Parameters
    ----------
    reallat : RealLattice
        Represents the crystal's lattice in real space
    l_atoms : sequence of BasisAtoms
        Represents the crystal's atom basis where each element represents
        a subset of basis atoms belonging to the same species
    """

    def __init__(self, reallat: RealLattice, l_atoms: list[BasisAtoms]):
        if not isinstance(reallat, RealLattice):
            raise TypeError(type_mismatch_msg("reallat", reallat, RealLattice))
        self.reallat: RealLattice = reallat
        """Represents the crystal's lattice in real space"""
        self.recilat: ReciLattice = ReciLattice.from_reallat(self.reallat)
        """Represents the crystal's lattice in reciprocal space"""

        for ityp, typ in enumerate(l_atoms):
            if not isinstance(typ, BasisAtoms):
                raise TypeError(
                    type_mismatch_msg(f"l_atoms[{ityp}]", l_atoms[ityp], BasisAtoms)
                )
            if typ.reallat is not self.reallat:
                raise ValueError(
                    obj_mismatch_msg(
                        f"l_atoms[{ityp}].reallat", typ.reallat, "reallat", reallat
                    )
                )
        self.l_atoms: list[BasisAtoms] = l_atoms
        """Represents the crystal's atom basis where each element represents
        a subset of basis atoms belonging to the same species"""
        self.symm: CrystalSymm = CrystalSymm(self)
        """Symmetry module of the Crystal"""

    @property
    def numel(self) -> int:
        """Total number of valence elecrons per unit cell in crystal"""
        return sum(sp.valence * sp.numatoms for sp in self.l_atoms)

    def gen_supercell(self, repeats: tuple[int, int, int]) -> Crystal:
        """Generates a supercell"""
        try:
            repeats = tuple(repeats)
            for ni in repeats:
                if not isinstance(ni, int) or ni < 0:
                    raise TypeError
        except TypeError as e:
            raise TypeError(
                type_mismatch_seq_msg("repeats", repeats, "positive integers")
            ) from e

        if len(repeats) != 3:
            raise ValueError(
                "'repeats' must contain 3 elements. " f"got {len(repeats)}"
            )

        xi = [np.arange(n, dtype="i8") for n in repeats]
        grid = np.array(
            np.meshgrid(*xi, indexing="ij"), like=self.reallat.latvec
        ).reshape((3, -1, 1))

        reallat = self.reallat
        alat_sup = repeats[0] * reallat.alat
        latvec_sup = np.array(repeats, like=reallat.latvec) * reallat.latvec
        reallat_sup = RealLattice(alat_sup, latvec_sup)
        l_atoms_sup = []
        for sp in self.l_atoms:
            r_cryst = (grid + sp.r_cryst.reshape((3, 1, -1))).reshape(3, -1)
            r_cart_sup = reallat.cryst2cart(r_cryst)
            r_cryst_sup = reallat_sup.cart2cryst(r_cart_sup)
            l_atoms_sup.append(
                BasisAtoms(sp.label, sp.ppdata, sp.mass, reallat_sup, r_cryst_sup)
            )

        return Crystal(reallat_sup, l_atoms_sup)

    def __repr__(self, indent="") -> str:
        res = "Crystal(\n    "+indent+f"reallat={self.reallat.__repr__(indent+'    ')}, \n    "+indent+f"l_atoms=["
        for sp in self.l_atoms:
            res += "\n"+indent + "  " + sp.__repr__(indent=indent+"    ")

        res += "\n    "+indent+"  ])"
        return res

    def __str__(self) -> str:
        
        alat_str = f"Lattice parameter 'alat' :   {self.reallat.alat:.5f}  a.u."
        cellvol_str = f"Unit cell volume         :  {self.reallat.cellvol:.5f}  (a.u.)^3"
        num_atoms_str = f"Number of atoms/cell     : {sum(sp.numatoms for sp in self.l_atoms)}"
        num_types_str = f"Number of atomic types   : {len(self.l_atoms)}"
        num_electrons_str = f"Number of electrons      : {self.numel}"
        
        reallat_str = str(self.reallat)
        atoms_str = ""
        for i, sp in enumerate(self.l_atoms, start=1):
            atoms_str += f"\n\nAtom Species #{i}\n{str(sp)}"

        return (
            f"{alat_str}\n"
            f"{cellvol_str}\n"
            f"{num_atoms_str}\n"
            f"{num_types_str}\n"
            f"{num_electrons_str}\n\n"
            f"{reallat_str}\n"
            f"{atoms_str}"
        )


class CrystalSymm:
    """Module for working with symmetries of given crystal"""

    symprec: float = 1e-5
    check_supercell: bool = True
    use_all_frac: bool = False

    def __init__(self, crystal: Crystal):
        assert isinstance(crystal, Crystal)
        self.crystal: Crystal = crystal
        lattice = crystal.reallat.latvec.T
        positions = [sp.r_cryst.T for sp in crystal.l_atoms]
        numbers = np.repeat(range(len(positions)), [len(pos) for pos in positions])
        positions = np.concatenate(positions, axis=0)

        if qtmconfig.gpu_enabled:
            reallat_symm = get_symmetry(
                (lattice.get(), positions.get(), numbers), symprec=self.symprec
            )
        else:
            reallat_symm = get_symmetry(
                (lattice, positions, numbers), symprec=self.symprec
            )
        del reallat_symm["equivalent_atoms"]
        if reallat_symm is None:
            reallat_symm = {
                "rotations": np.eye(3, dtype="i4").reshape((1, 3, 3)),
                "translations": np.zeros(3, dtype="f8"),
            }

        if self.check_supercell:
            idx_identity = np.nonzero(
                np.all(reallat_symm["rotations"] == np.eye(3, dtype="i8"), axis=(1, 2))
            )[0]
            if len(idx_identity) != 1:
                idx_notrans = np.nonzero(
                    np.linalg.norm(reallat_symm["translations"], axis=1) <= self.symprec
                )[0]
                for k, v in reallat_symm.items():
                    reallat_symm[k] = v[idx_notrans]

        recilat_symm = np.linalg.inv(
            reallat_symm["rotations"].transpose((0, 2, 1))
        ).astype("i4")

        numsymm = len(reallat_symm["rotations"])
        self.symm: np.ndarray = np.array(
            [
                (
                    reallat_symm["rotations"][i],
                    reallat_symm["translations"][i],
                    recilat_symm[i],
                )
                for i in range(numsymm)
            ],
            dtype=[
                ("reallat_rot", "i4", (3, 3)),
                ("reallat_trans", "f8", (3,)),
                ("recilat_rot", "i4", (3, 3)),
            ],
        )
        """List of Symmetry operations of input crystal"""

    @property
    def numsymm(self) -> int:
        """Total number of crystal symmetries"""
        return len(self.symm)

    @property
    def reallat_rot(self):
        return self.symm["reallat_rot"]

    @property
    def reallat_trans(self):
        return self.symm["reallat_trans"]

    @property
    def recilat_rot(self):
        return self.symm["recilat_rot"]

    def filter_frac_trans(self, grid_shape: tuple[int, int, int]):
        if self.use_all_frac:
            return

        fac = np.multiply(self.symm["reallat_trans"], grid_shape)
        idx_comm = np.nonzero(
            np.linalg.norm(fac - np.rint(fac), axis=1) <= self.symprec
        )[0]
        self.symm = self.symm[idx_comm].copy()

    @property
    def correspondence_table(self):
        symm_tol=6
        """In this function we calculate a correspondence table 
        between the original coordinates and the transformed coordinates. 
        In this table, the [i,j] the component represents the index of the
        transformed coordinates corresponding to the original coordinates.
        """
        cryst=self.crystal
        l_atoms=cryst.l_atoms
        coords_cryst_all = np.concatenate([sp.r_cryst for sp in l_atoms], axis=1)
        tot_num=coords_cryst_all.shape[1]

        #Getting the Transformation Matrices
        reallat_rot=self.reallat_rot
        reallat_trans=self.reallat_trans

        #Applying the transformations-rotation
        coords_cryst_rot=(reallat_rot @ coords_cryst_all)
        coords_cryst_rot=np.swapaxes(coords_cryst_rot, 1, 2)

        #Applying the transformations-translation
        coords_cryst_transormed= coords_cryst_rot + self.reallat_trans.reshape(-1, 1, 3)

        #Number of rot-trans symmetry operations in the crystal
        num_rot_trans=self.reallat_trans.shape[0]

        ##Comparing with the original coordinates
        correspondence_table=-np.ones((num_rot_trans, tot_num))

        ##Truncating the transformed coordinates
        coords_cryst_transormed=np.round(coords_cryst_transormed, symm_tol)
        coords_cryst_all=np.round(coords_cryst_all, symm_tol)

        

        for i in range(num_rot_trans):
            coords_transformed=coords_cryst_transormed[i]
            for j in range(tot_num):
                coord_original=coords_cryst_all[:, j]
                for k in range(tot_num):
                    coord_transformed=coords_transformed[k]
                    diff=coord_original-coord_transformed
                    if np.all(diff==np.floor(diff)):
                        correspondence_table[i, j]=k
                        break
        
        return correspondence_table.astype(np.int16)
                

    def symmetrize_vec(self, vec):
        """If every atom in the lattice has some vectors associated with it, then this function,
        symmetrises the vectors. The input is a Nx3 array where N is the number of atoms.
        The vectors are in cartesian coordinates."""
        cryst=self.crystal
        l_atoms = cryst.l_atoms
        tot_num = np.sum([sp.numatoms for sp in l_atoms])

        if not isinstance(vec, np.ndarray):
            raise TypeError(type_mismatch_msg('vec', vec, np.ndarray))
        if vec.ndim != 2:
            raise ValueError(f"Expected 2D array, got {vec.ndim}D array")
        if vec.shape != (tot_num, 3):
            raise ValueError(f"Expected shape {(tot_num,3)}, got {vec.shape}")

        ##Converting the vector into crystal coordinates
        vec=cryst.reallat.cart2cryst(vec, axis=1)
    
        #Getting the Correspondence Table
        correspondence_table=self.correspondence_table

        ##Getting the Inverses of the Rotation Matrices
        rot_inv=np.array([np.linalg.inv(self.reallat_rot).T for rot in self.reallat_rot])

        #Total number of trans_rot symmetry operations
        num_trans_rot=self.reallat_trans.shape[0]

        #Symmetrizing the Vectors
        vec_sym=np.zeros_like(vec)

        for atom in range(tot_num):
            icounter=0
            for isymm in range(num_trans_rot):
                atom_transformed=correspondence_table[isymm, atom]
                if atom_transformed!=-1:
                    vec_sym[atom]=vec_sym[atom]+self.reallat_rot[isymm] @ vec[atom_transformed]
                    icounter+=1
                    #print(f"the atom {atom} is transformed to {atom_transformed} by {itrans}th symmetry operation")
                    #print(f"the vector of atom {atom_transformed} is {vec[atom_transformed]}, transformed to {rot_inv[itrans] @ vec[atom_transformed]} by {itrans}th symmetry operation")
            vec_sym[atom]/=icounter

        #Converting back to cartesian coordinates
        vec_sym=cryst.reallat.cryst2cart(vec_sym, axis=1)
        return vec_sym


    def symmetrize_matrix(self, matrix):
        """The total system has a matrix associated with it. This function symmetrizes the matrix."""
        cryst=self.crystal
        l_atoms = cryst.l_atoms
        tot_num = np.sum([sp.numatoms for sp in l_atoms])

        if not isinstance(matrix, np.ndarray):
            raise TypeError(type_mismatch_msg('matrix', matrix, np.ndarray))
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D array, got {matrix.ndim}D array")
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected shape {(3,3)}, got {matrix.shape}")

        ##ritting down the primitive vectors
        primvec=cryst.reallat.primvec
        primvec_inv=cryst.reallat.primvec_inv

        ##Conveting to crystal coordinates
        matrix= primvec_inv@matrix@primvec_inv.T


        '''##Getting the Inverses of the Rotation Matrices
        rot_inv=np.array([np.linalg.inv(rot).T for rot in self.reallat_rot])'''

        #Total number of trans_rot symmetry operations
        num_rot_trans=self.reallat_trans.shape[0]

        #Symmetrizing the Matrix
        matrix_sym=np.zeros_like(matrix)

        for isym in range(num_rot_trans):
            matrix_sym+=self.reallat_rot[isym] @ matrix @ self.reallat_rot[isym].T
            '''print("-------------------------------------------------")
            print("matrix before", matrix)
            print("s=", self.reallat_rot[isym])
            print("mmatrix after", self.reallat_rot[isym] @ matrix @ self.reallat_rot[isym].T)'''
        
        matrix_sym/=num_rot_trans

        ##Converting back to Cartesian Coordinates
        matrix_sym=primvec@matrix_sym@primvec.T

        return matrix_sym

