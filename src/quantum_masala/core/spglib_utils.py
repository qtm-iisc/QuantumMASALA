import numpy as np
import spglib

from quantum_masala import config
from quantum_masala.core import Crystal
from quantum_masala.core.lattice import Lattice

symm_dtype = [("rotations", "i4", (3, 3)), ("translations", "f8", (3,))]


def compute_symmetry_spglib(
    spglib_cell: tuple[np.ndarray, np.ndarray, tuple[int, ...]]
) -> np.ndarray:
    """Wrapper to call `spglib.get_symmetry` for determining symmetry of given crystal

    Parameters
    ----------
    spglib_cell: tuple[np.ndarray, np.ndarray, tuple[int, ...]]
        A tuple containing crystal structure information supported by Spglib >= 1.9.1.

    Returns
    -------
    A Structured NumPy array containing a list of symmetry operations represented by
    a rotation matrix and a corresponding fractional translation vectors,
    `rotations` and `translations` respectively.

    Notes
    -----
    Please refer to `Spglib for Python` page for information regarding the `spglib_cell`:
    https://spglib.github.io/spglib/python-spglib.html#crystal-structure-cell
    """
    symm_data = spglib.get_symmetry(spglib_cell, symprec=config.spglib_symprec)
    if symm_data is None:
        symm_data = {
            "rotations": np.eye(3, dtype="i4"),
            "translations": np.zeros(3, dtype="f8"),
        }

    numsymm = symm_data["rotations"].shape[0]

    symmetry = np.array(
        [
            (symm_data["rotations"][i], symm_data["translations"][i])
            for i in range(numsymm)
        ],
        dtype=symm_dtype,
    )
    return symmetry


def get_symmetry_lattice(lat: Lattice) -> np.ndarray:
    """Function for determining symmetry operations of given lattice"""
    lattice = np.transpose(lat.primvec)
    positions = np.zeros((1, 3))
    numbers = (0,)

    cell = (lattice, positions, numbers)
    return compute_symmetry_spglib(cell)


def get_symmetry_crystal(crystal: Crystal,
                         check_supercell: bool = True) -> (np.ndarray, np.ndarray):
    """Function for determining symmetry operations of given crystal.
    Returns symmetries of both real-space and reciprocal-space lattices"""
    spglib_cell = crystal.spglib_cell

    reallat_symm = compute_symmetry_spglib(spglib_cell)
    recilat_symm = None
    if reallat_symm is not None:
        recilat_symm = np.linalg.inv(
            np.transpose(reallat_symm["rotations"], axes=(0, 2, 1))
        ).astype("i4")

    if not check_supercell:
        return reallat_symm, recilat_symm

    idx_idsymm = np.nonzero(
        np.all(reallat_symm["rotations"] - np.eye(3, dtype='i8'), axis=(1, 2))
    )
    if len(idx_idsymm[0]) == 1:
        return reallat_symm, recilat_symm

    idx_notrans = np.nonzero(
        np.linalg.norm(reallat_symm["translations"], axis=1) <= config.spglib_symprec
    )
    return reallat_symm[idx_notrans], recilat_symm[idx_notrans]