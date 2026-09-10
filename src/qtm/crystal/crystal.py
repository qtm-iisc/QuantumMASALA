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
            # 'sp.ppdata' is None whenever 'sp' was built with a bare
            # valence-electron-count int instead of a real pseudopotential
            # (BasisAtoms.__init__ keeps only 'sp.valence' in that case) --
            # passing it through unchanged would silently set the new
            # species' valence to -1 (BasisAtoms.__init__'s ppdata=None
            # default) instead of preserving the original count.
            ppdata_sup = sp.ppdata if sp.ppdata is not None else sp.valence
            l_atoms_sup.append(
                BasisAtoms(sp.label, ppdata_sup, sp.mass, reallat_sup, r_cryst_sup)
            )

        return Crystal(reallat_sup, l_atoms_sup)

    def __repr__(self, indent="") -> str:
        res = (
            "Crystal(\n    "
            + indent
            + f"reallat={self.reallat.__repr__(indent+'    ')}, \n    "
            + indent
            + f"l_atoms=["
        )
        for sp in self.l_atoms:
            res += "\n" + indent + "  " + sp.__repr__(indent=indent + "    ")

        res += "\n    " + indent + "  ])"
        return res

    def __str__(self) -> str:
        alat_str = f"Lattice parameter 'alat' :   {self.reallat.alat:.5f}  a.u."
        cellvol_str = (
            f"Unit cell volume         :  {self.reallat.cellvol:.5f}  (a.u.)^3"
        )
        num_atoms_str = (
            f"Number of atoms/cell     : {sum(sp.numatoms for sp in self.l_atoms)}"
        )
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

    _KPOINT_ROUND_PREC: int = 6

    @staticmethod
    def _find_symm(
        recilat_rot: np.ndarray,
        k_src: np.ndarray,
        k_dest: np.ndarray,
        time_reversal: bool = None,
    ) -> list[tuple[int, bool]]:
        tol = 10.0 ** (-CrystalSymm._KPOINT_ROUND_PREC)
        matches = []
        for tr in (False, True) if time_reversal is None else (time_reversal,):
            sign = -1 if tr else 1
            k_rot = sign * np.tensordot(recilat_rot, k_src, axes=1)
            g0 = np.rint(k_rot - k_dest)
            match = np.all(np.abs(k_rot - g0 - k_dest) < tol, axis=-1)
            matches.extend((int(i), tr) for i in np.nonzero(match)[0])
        return matches

    def find_symm(
        self,
        k_src: tuple[float, float, float],
        k_dest: tuple[float, float, float],
        time_reversal: bool = None,
    ) -> list[tuple[int, bool]]:
        r"""Lists every symmetry operation relating `k_src` to `k_dest`.

        Since more than one ``(isymm, time_reversal)`` pair can satisfy
        :math:`\mathbf{k}_{dest} = \pm S_{isymm}\mathbf{k}_{src}
        \pmod{\mathbf{G}}` (e.g. when `k_src` has a nontrivial little
        group), use this to see every candidate rather than assuming there
        is only one.

        Parameters
        ----------
        k_src, k_dest : tuple[float, float, float]
            The two k-points, in crystal coordinates.
        time_reversal : bool, optional
            If given, only operations combined (`True`) or not combined
            (`False`) with time reversal are searched for. If `None`
            (default), both are tried.

        Returns
        -------
        list[tuple[int, bool]]
            Every ``(isymm, time_reversal)`` pair, in ascending order of
            `isymm` (operations without time reversal listed before those
            with it), such that symmetry operation ``isymm`` of this
            crystal (combined with time reversal if `time_reversal` is
            `True`) relates `k_src` to `k_dest`.
        """
        k_src = np.around(np.asarray(k_src, dtype="f8"), self._KPOINT_ROUND_PREC)
        k_dest = np.around(np.asarray(k_dest, dtype="f8"), self._KPOINT_ROUND_PREC)
        return self._find_symm(self.recilat_rot, k_src, k_dest, time_reversal)

    def little_group(
        self,
        k_cryst: tuple[float, float, float],
        time_reversal: bool = None,
    ) -> list[tuple[int, bool]]:
        r"""Lists every symmetry operation that fixes `k_cryst` (mod a
        reciprocal lattice vector) -- its little group (also called its
        stabilizer, or small group).

        This is exactly ``find_symm(k_cryst, k_cryst, time_reversal)``: an
        operation fixes `k_cryst` iff it relates `k_cryst` to itself. A
        k-point with a nontrivial little group (more than just the
        identity, for `time_reversal=False`) can have symmetry-protected
        degenerate bands there; the little group's irreducible
        representations give the degeneracy dimensions that are actually
        allowed (see e.g. the `spgrep
        <https://github.com/spglib/spgrep>`_ package for computing those
        from a crystal structure -- this method only finds the *operations*
        forming the little group, not their representations).

        Parameters
        ----------
        k_cryst : tuple[float, float, float]
            The k-point, in crystal coordinates.
        time_reversal : bool, optional
            If given, only operations combined (`True`) or not combined
            (`False`) with time reversal are searched for. If `None`
            (default), both are tried.

        Returns
        -------
        list[tuple[int, bool]]
            Every ``(isymm, time_reversal)`` pair fixing `k_cryst`, in the
            same format as `find_symm`.
        """
        return self.find_symm(k_cryst, k_cryst, time_reversal)
