# TODO: Validate all ibrav
from typing import Optional
import numpy as np

from qtm.constants import ANGSTROM


def cellparam2latvec(
    option: str, cellparam: np.ndarray, celldm: dict[int, float], a: Optional[float]
):
    """

    Parameters
    ----------
    cellparam : List[float]
        Fields in 'CELL_PARAMETERS' namelist in QE input files.
        Represents the lattice vectors in cartesian basis
    option : str
        'CELL_PARAMETERS' card option; Units of the lattice
        vectors specified in cellparam

    Returns
    -------
    alat: float
        Lattice Parameter 'a' in Bohr
    latvec: np.ndarray
        Array containing Lattice vectors in Bohr; Each column
        corresponds to a lattice vector
    """
    cellparam = np.array(cellparam)
    alat = None
    if option == "angstrom":
        cellparam *= ANGSTROM
    elif option == "alat":
        celldm_1 = celldm[1]
        if celldm_1 is None and a is None:
            raise ValueError(
                "for 'CELL PARAMETERS' option 'alat', "
                "either 'celldm(1)' or 'a' need to be specified"
            )
        if celldm_1 is None:
            alat = a * ANGSTROM
        else:
            alat = celldm_1

    latvec = np.array(np.transpose(cellparam), order="C")
    if option == "alat":
        latvec *= alat
    else:
        alat = np.linalg.norm(latvec[:, 0])
    return alat, latvec


def trad2celldm(
    ibrav: int, a: float, b: float, c: float, cosab: float, cosbc: float, cosac: float
):
    celldm = {1: a * ANGSTROM}

    if ibrav in [8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14]:
        if b is None:
            raise ValueError(f"'b' must be specified for ibrav={ibrav}")
        celldm[2] = b / a

    if ibrav in [4, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14]:
        if c is None:
            raise ValueError(f"'c' must be specified for ibrav={ibrav}")
        celldm[3] = c / a

    if ibrav in [5, -5, 12, 13]:
        if cosab is None:
            raise ValueError(f"'cosab' must be specified for ibrav={ibrav}")
        celldm[4] = cosab  # cos(gamma) = cosAB

    elif ibrav in [
        14,
    ]:
        if cosbc is None:
            raise ValueError(f"'cosbc' must be specified for ibrav={ibrav}")
        celldm[4] = cosbc  # cos(alpha) = cosBC

    if ibrav in [-12, -13, 14]:
        if cosac is None:
            raise ValueError(f"'cosac' must be specified for ibrav={ibrav}")
        celldm[5] = cosac  # cos(beta) = cosAC

    if ibrav in [14]:
        if cosab is None:
            raise ValueError(f"'cosab' must be specified for ibrav={ibrav}")
        celldm[6] = cosab

    return celldm


def basis_cubic_p(celldm):  # ibrav = 1
    a = celldm[1]
    v1 = [a, 0, 0]
    v2 = [0, a, 0]
    v3 = [0, 0, a]
    return [v1, v2, v3]


def basis_cubic_f(celldm):  # ibrav = 2
    aby2 = celldm[1] / 2
    v1 = [-aby2, 0, aby2]
    v2 = [0, aby2, aby2]
    v3 = [-aby2, aby2, 0]
    return [v1, v2, v3]


def basis_cubic_i(celldm):  # ibrav = 3
    aby2 = celldm[1] / 2
    v1 = [aby2, aby2, aby2]
    v2 = [-aby2, aby2, aby2]
    v3 = [-aby2, -aby2, aby2]
    return [v1, v2, v3]


def basis_cubic_i_symm(celldm):  # ibrav = -3
    aby2 = celldm[1] / 2
    v1 = [-aby2, aby2, aby2]
    v2 = [aby2, -aby2, aby2]
    v3 = [aby2, aby2, -aby2]
    return [v1, v2, v3]


def basis_hexagonal(celldm):  # ibrav = 4
    a = celldm[1]
    c = a * celldm[3]  # celldm[3] = c/a
    v1 = [a, 0, 0]
    v2 = [-a / 2, a * np.sqrt(3) / 2, 0]
    v3 = [0, 0, c]
    return [v1, v2, v3]


def basis_trigonal_r_c(celldm):  # ibrav = 5
    a = celldm[1]
    cosg = celldm[4]  # celldm[4] = cos(gamma) = cosg

    tx = np.sqrt((1 - cosg) / 2)
    ty = np.sqrt((1 - cosg) / 6)
    tz = np.sqrt((1 + 2 * cosg) / 3)

    v1 = [a * tx, -a * ty, a * tz]
    v2 = [0, 2 * a * ty, a * tz]
    v3 = [-a * tx, -a * ty, a * tz]
    return [v1, v2, v3]


def basis_trigonal_r_111(celldm):  # ibrav = -5
    a = celldm[1]
    cosg = celldm[4]  # celldm[4] = cos(gamma) = cosg

    ty = np.sqrt((1 - cosg) / 6)
    tz = np.sqrt((1 + 2 * cosg) / 3)

    aa = a / np.sqrt(3)
    u = tz - 2 * np.sqrt(2) * ty
    v = tz + np.sqrt(2) * ty

    v1 = [aa * u, aa * v, aa * v]
    v2 = [aa * v, aa * u, aa * v]
    v3 = [aa * v, aa * v, aa * u]
    return [v1, v2, v3]


def basis_tetragonal_p(celldm):  # ibrav = 6
    a = celldm[1]
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a, 0, 0]
    v2 = [0, a, 0]
    v3 = [0, 0, c]
    return [v1, v2, v3]


def basis_tetragonal_i(celldm):  # ibrav = 7
    a = celldm[1]
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a / 2, -a / 2, c / 2]
    v2 = [a / 2, a / 2, c / 2]
    v3 = [-a / 2, -a / 2, c / 2]
    return [v1, v2, v3]


def basis_orthorhombic_p(celldm):  # ibrav = 8
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a, 0, 0]
    v2 = [0, b, 0]
    v3 = [0, 0, c]
    return [v1, v2, v3]


def basis_orthorhombic_bco(celldm):  # ibrav = 9
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a / 2, b / 2, 0]
    v2 = [-a / 2, b / 2, 0]
    v3 = [0, 0, c]
    return [v1, v2, v3]


def basis_orthorhombic_bco_alt(celldm):  # ibrav = -9
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a / 2, -b / 2, 0]
    v2 = [a / 2, b / 2, 0]
    v3 = [0, 0, c]
    return [v1, v2, v3]


def basis_orthorhombic_bca(celldm):  # ibrav = 91
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a, 0, 0]
    v2 = [0, b / 2, -c / 2]
    v3 = [0, b / 2, c / 2]
    return [v1, v2, v3]


def basis_orthorhombic_face(celldm):  # ibrav = 10
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a / 2, 0, c / 2]
    v2 = [a / 2, b / 2, 0]
    v3 = [0, b / 2, c / 2]
    return [v1, v2, v3]


def basis_orthorhombic_body(celldm):  # ibrav = 11
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    v1 = [a / 2, b / 2, c / 2]
    v2 = [-a / 2, b / 2, c / 2]
    v3 = [-a / 2, -b / 2, c / 2]
    return [v1, v2, v3]


def basis_monoclinic_p_c(celldm):  # ibrav = 12
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    cosAB = celldm[4]  # celldm[4] = cos(ab)
    sinAB = np.sqrt(1 - cosAB**2)
    v1 = [a, 0, 0]
    v2 = [b * cosAB, b * sinAB, 0]
    v3 = [0, 0, c]
    return [v1, v2, v3]


def basis_moniclinic_p_b(celldm):  # ibrav = -12
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    cosAC = celldm[5]  # celldm[5] = cos(ac)
    sinAC = np.sqrt(1 - cosAC**2)
    v1 = [a, 0, 0]
    v2 = [0, b, 0]
    v3 = [c * cosAC, 0, c * sinAC]
    return [v1, v2, v3]


def basis_monoclinic_base_c(celldm):  # ibrav = 13
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[1] = b/a
    c = celldm[3] * celldm[1]  # celldm[2] = c/a
    cosg = celldm[4]  # celldm[4] = cos(gamma)
    sing = np.sqrt(1 - cosg**2)
    v1 = [a / 2, 0, -c / 2]
    v2 = [b * cosg, b * sing, 0]
    v3 = [a / 2, 0, c / 2]
    return [v1, v2, v3]


def basis_monoclinic_base_b(celldm):  # ibrav = -13
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[1] = b/a
    c = celldm[3] * celldm[1]  # celldm[2] = c/a
    cosb = celldm[5]  # celldm[5] = cos(beta)
    sinb = np.sqrt(1 - cosb**2)
    v1 = [a / 2, b / 2, 0]
    v2 = [-a / 2, b / 2, 0]
    v3 = [c * cosb, 0, c * sinb]
    return [v1, v2, v3]


def basis_triclinic(celldm):  # ibrav = 14
    a = celldm[1]
    b = celldm[2] * celldm[1]  # celldm[2] = b/a
    c = celldm[3] * celldm[1]  # celldm[3] = c/a
    cosBC = celldm[4]  # celldm[4] = cos(bc)
    cosAC = celldm[5]  # celldm[5] = cos(ac)
    cosAB = celldm[6]  # celldm[6] = cos(ab)
    sinAB = np.sqrt(1 - cosAB**2)
    v1 = [a, 0, 0]
    v2 = [b * cosAB, b * sinAB, 0]
    v3 = [
        c * cosAC,
        c * (cosBC - cosAC * cosAB) / sinAB,
        c
        * np.sqrt(1 + 2 * cosBC * cosAC * cosAB - cosAB**2 - cosAC**2 - cosAB**2)
        / sinAB,
    ]
    return [v1, v2, v3]


ibrav_list = {
    1: basis_cubic_p,
    2: basis_cubic_f,
    3: basis_cubic_i,
    -3: basis_cubic_i_symm,
    4: basis_hexagonal,
    5: basis_trigonal_r_c,
    -5: basis_trigonal_r_111,
    6: basis_tetragonal_p,
    7: basis_tetragonal_i,
    8: basis_orthorhombic_p,
    9: basis_orthorhombic_bco,
    -9: basis_orthorhombic_bco_alt,
    91: basis_orthorhombic_bca,
    10: basis_orthorhombic_face,
    11: basis_orthorhombic_body,
    12: basis_monoclinic_p_c,
    -12: basis_moniclinic_p_b,
    13: basis_monoclinic_base_c,
    -13: basis_monoclinic_base_b,
    14: basis_triclinic,
}


def ibrav2latvec(ibrav: int, celldm: dict[int, float]):
    alat = celldm[1]

    if ibrav in ibrav_list:
        try:
            v1, v2, v3 = ibrav_list[ibrav](celldm)
        except:
            raise ValueError(
                f"'ibrav2latvec' routine failed for the following parameters: 'ibrav'={ibrav}\n"
                f"celldm(1)={celldm[1]}  celldm(2)={celldm[2]}  celldm(3)={celldm[3]}\n"
                f"celldm(4)={celldm[4]}  celldm(5)={celldm[5]}  celldm(6)={celldm[6]}"
            )
        latvec = np.array([v1, v2, v3]).transpose()
        return alat, np.array(latvec, order="C")
