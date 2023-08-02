# from __future__ import annotations
__all__ = ["UPFv2Data"]

import xml.etree.ElementTree as ET

import copy
import numpy as np
from dataclasses import dataclass
from qtm.typing import Optional

from qtm.crystal.basis_atoms import PseudoPotFile
from qtm.constants import RYDBERG

_LIBXC_MAP = {
    "pbe": ("gga_x_pbe", "gga_c_pbe")
}


@dataclass
class UPFv2Data(PseudoPotFile):
    r"""Container to store data from UPF Files v.2.0.1.

    All quantities read from file are converted to Hartree Atomic Units.
    Implementation based on UPF Specification given in:
    http://pseudopotentials.quantum-espresso.org/home/unified-pseudopotential-format.

    Notes
    -----
    As `QuantumMASALA does not support PAW or Ultrasoft Pseudopotentials, they are omitted in this
    implementation.
    """

    # Fields in 'PP_HEADER'.
    generated: str
    author: str
    date: str
    comment: str

    element: str
    pseudo_type: str
    relativistic: str
    is_ultrasoft: bool
    is_paw: bool
    is_coulomb: bool
    has_so: bool
    has_wfc: bool
    core_correction: bool
    functional: str  # NOTE: Value stored is libxc equivalent of the one in file
    z_valence: float
    total_psenergy: float
    wfc_cutoff: float
    rho_cutoff: float
    l_max: int
    l_local: int
    mesh_size: int
    number_of_wfc: int
    number_of_proj: int

    # Fields in 'PP_MESH'.
    r: np.ndarray
    r_ab: np.ndarray

    # Fields in 'PP_NLCC'.
    rho_atc: Optional[np.ndarray]

    # Fields in 'PP_LOCAL'.
    vloc: np.ndarray

    # Fields in 'PP_NONLOCAL' (Parsed and ordered into lists; Ultrasoft not implemented).
    l_kb_rbeta: list[np.ndarray]
    l_kb_l: list[int]
    dij: np.ndarray

    # Fields in 'PP_RHOATOM'.
    rhoatom: np.ndarray

    @classmethod
    def from_file(cls, label: str, dirname: str):
        """Factory Method to parse UPFv2 files into `UPFv2Data` instances.

        Parameters
        ----------
        label : str
            Label of the atom type.
        dirname : str
            Path of the input file.
        """
        data = copy.deepcopy(cls.__annotations__)

        tree = ET.parse(dirname)
        root = tree.getroot()

        # Reading mandatory fields 'PP_HEADER', 'PP_MESH', 'PP_LOCAL'
        for child in root:
            if child.tag == "PP_HEADER":
                for key, val in child.attrib.items():
                    if key in data:
                        typ = data[key]
                        if typ == bool:
                            data[key] = val.lower() == "t"
                        else:
                            data[key] = typ(val)

            elif child.tag == "PP_MESH":
                for gchild in child:
                    if gchild.tag == "PP_R":
                        data["r"] = np.array(gchild.text.split(), dtype=np.float64)
                    elif gchild.tag == "PP_RAB":
                        data["r_ab"] = np.array(gchild.text.split(), dtype=np.float64)

            elif child.tag == "PP_LOCAL":
                data["vloc"] = np.array(child.text.split(), dtype=np.float64) * RYDBERG

            elif child.tag == "PP_RHOATOM":
                data["rhoatom"] = np.array(
                    child.text.split(), dtype=np.float64
                )  # TODO: Check units

        # Reading Optional field 'PP_NLCC'
        if data["core_correction"] is True:
            child = root.findall("PP_NLCC")[0]
            data["rho_atc"] = np.array(
                child.text.split(), dtype=np.float64
            )  # TODO: Check units
        else:
            data["rho_atc"] = None

        # Reading field 'PP_NONLOCAL'
        if data["number_of_proj"] != 0:
            child = root.findall("PP_NONLOCAL")[0]

            l_kb_l = []
            l_kb_rbeta = []
            dij = None
            n_beta = 0
            for gchild in child:
                if gchild.tag == "PP_DIJ":
                    dij = np.array(gchild.text.split(), dtype=np.float64) * RYDBERG
                elif gchild.tag.startswith("PP_BETA."):
                    n_beta += 1
                    l = int(gchild.attrib["angular_momentum"])
                    beta = np.array(gchild.text.split(), dtype=np.float64)
                    l_kb_l.append(l)
                    l_kb_rbeta.append(beta)

            data["dij"] = dij.reshape(n_beta, n_beta)
            data["l_kb_l"] = l_kb_l
            data["l_kb_rbeta"] = l_kb_rbeta
        else:
            data["dij"] = np.zeros((0, 0), dtype=np.float64)
            data["l_beta_times_r"] = []

        data['libxc_func'] = None
        funcname = data['functional']
        if funcname.lower() in _LIBXC_MAP:
            data['libxc_func'] = _LIBXC_MAP[funcname.lower()]

        return cls(dirname, data['z_valence'], **data)
