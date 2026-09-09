"""Simple, DFT-free unit tests for `qtm.pseudo.upf.UPFv2Data`: parsing
correctness against the raw XML, unit conversions (`RYDBERG` applied to
`vloc`/`dij`), and `PseudoPotFile.__post_init__`'s filename/md5 bookkeeping.

Uses the existing 'Si_ONCV_PBE-1.2.upf' file already present in
'tests/system_tests/' -- pure XML parsing and arithmetic, no crystal, no SCF.
"""
import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

from qtm.constants import RYDBERG
from qtm.pseudo import UPFv2Data

UPF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "system_tests", "Si_ONCV_PBE-1.2.upf"
)

data = UPFv2Data.from_file(UPF_PATH)

_tree = ET.parse(UPF_PATH)
_root = _tree.getroot()
_header = {c.tag: dict(c.attrib) for c in _root if c.tag == "PP_HEADER"}["PP_HEADER"]


def test_header_scalar_fields_match_raw_xml():
    assert data.element == _header["element"]
    assert data.functional == _header["functional"]
    assert data.pseudo_type == _header["pseudo_type"]
    assert np.isclose(data.z_valence, float(_header["z_valence"]))
    assert data.valence == int(np.rint(float(_header["z_valence"])))
    assert data.l_max == int(_header["l_max"])
    assert data.l_local == int(_header["l_local"])
    assert data.mesh_size == int(_header["mesh_size"])
    assert data.number_of_proj == int(_header["number_of_proj"])
    assert data.number_of_wfc == int(_header["number_of_wfc"])
    assert np.isclose(data.rho_cutoff, float(_header["rho_cutoff"]))


def test_header_boolean_fields_match_raw_xml():
    assert data.is_ultrasoft == (_header["is_ultrasoft"].lower() == "t")
    assert data.is_paw == (_header["is_paw"].lower() == "t")
    assert data.is_coulomb == (_header["is_coulomb"].lower() == "t")
    assert data.has_so == (_header["has_so"].lower() == "t")
    assert data.core_correction == (_header["core_correction"].lower() == "t")


def test_wfc_cutoff_absent_from_header_parses_to_none():
    # This file's PP_HEADER has no 'wfc_cutoff' attribute at all (only
    # 'rho_cutoff' is present) -- regression test for a bug where an absent
    # *optional* header attribute was left as the bare type annotation
    # object (the class 'float' itself) instead of a sensible value.
    assert "wfc_cutoff" not in _header
    assert data.wfc_cutoff is None


def test_libxc_func_mapped_from_functional():
    assert data.functional == "PBE"
    assert data.libxc_func == ("gga_x_pbe", "gga_c_pbe")


def test_vloc_and_dij_convert_rydberg_to_hartree():
    for child in _root:
        if child.tag == "PP_LOCAL":
            raw_vloc = np.array(child.text.split(), dtype=np.float64)
        elif child.tag == "PP_NONLOCAL":
            for gchild in child:
                if gchild.tag == "PP_DIJ":
                    raw_dij = np.array(gchild.text.split(), dtype=np.float64)

    assert np.allclose(data.vloc, raw_vloc * RYDBERG)
    assert np.allclose(data.dij.reshape(-1), raw_dij * RYDBERG)
    assert np.isclose(RYDBERG, 0.5)  # Ry -> Ha conversion factor, sanity check


def test_dij_and_beta_projectors_consistent_with_number_of_proj():
    assert data.dij.shape == (data.number_of_proj, data.number_of_proj)
    assert len(data.l_kb_l) == data.number_of_proj
    assert len(data.l_kb_rbeta) == data.number_of_proj

    n_beta_children = sum(
        1
        for child in _root
        if child.tag == "PP_NONLOCAL"
        for gchild in child
        if gchild.tag.startswith("PP_BETA.")
    )
    assert n_beta_children == data.number_of_proj


def test_mesh_and_rhoatom_lengths_match_mesh_size():
    assert len(data.r) == data.mesh_size
    assert len(data.r_ab) == data.mesh_size
    assert len(data.rhoatom) == data.mesh_size
    assert len(data.vloc) == data.mesh_size


def test_filename_and_md5_checksum():
    assert data.filename == os.path.basename(UPF_PATH)
    expected_md5 = hashlib.md5()
    with open(UPF_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            expected_md5.update(chunk)
    assert data.md5_checksum == expected_md5.hexdigest()
