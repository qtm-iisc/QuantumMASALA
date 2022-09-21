import numpy as np
import spglib

from ..cryst import Lattice, Crystal

from quantum_masala.config import SPGLIB_CONFIG
SYMPREC = SPGLIB_CONFIG["SYMPREC"]
ANGLE_TOL = SPGLIB_CONFIG["ANGLE_TOL"]

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
    symm_data = spglib.get_symmetry(spglib_cell, symprec=SYMPREC)
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


def get_symmetry_crystal(crystal: Crystal) -> (np.ndarray, np.ndarray):
    """Function for determining symmetry operations of given crystal.
    Returns symmetries of both real-space and reciprocal-space lattices"""
    spglib_cell = crystal.spglib_cell

    reallat_symm = compute_symmetry_spglib(spglib_cell)
    recilat_symm = None
    if reallat_symm is not None:
        recilat_symm = np.linalg.inv(
            np.transpose(reallat_symm["rotations"], axes=(0, 2, 1))
        ).astype("i4")

    return reallat_symm, recilat_symm
