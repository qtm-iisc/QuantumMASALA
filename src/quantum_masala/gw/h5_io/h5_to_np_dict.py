# Reading WFN.h5 file to dict, with appropriate conversion and preparation


import h5py
import numpy as np


def read_input_h5(filename, wfn_prepare=True):
    """Read wfn.h5 to dict"""
    header = h5py.File(filename, "r")
    data = retrieve_dict(header)
    if wfn_prepare:
        prepare_data(data)  # Makes it specific to wfn.h5 and epsilon.py
    return data


def retrieve_dict(source):
    """
    Recursive function that reads nested HDF5 Groups to nested Dicts
    Datasets are converted to np.array
    """
    target = {}
    for sub in list(source.keys()):
        if isinstance(source[sub], h5py.Dataset):
            target[sub] = np.array(
                source[sub]
            )  # source[sub].dtype #to print dict of datatypes
        elif isinstance(source[sub], h5py.Group):
            target[sub] = retrieve_dict(source[sub])
    return target


def prepare_data(data):
    """Conversion of h5 data to numpy, Specific to wfn.h5"""

    kpoints = data["mf_header"]["kpoints"]
    wfns = data["wfns"]

    if kpoints["nspin"] * kpoints["nspinor"] == 1:
        for key in ["el", "ifmin", "ifmax", "occ"]:
            # Get rid of spin-spinor column, i.e. 1st column
            kpoints[key] = np.array(kpoints[key][0])
    wfns["coeffs"], wfns["gvecs"] = parse_wfns(
        wfns["coeffs"], wfns["gvecs"], kpoints["ngk"]
    )
    return


def parse_wfns(coeffs, gvecs, ngk):
    indices = np.cumsum(ngk)[:-1]
    print(coeffs.shape)
    complex_coeffs = coeffs[:, 0, :, 1] + 1j * coeffs[:, 0, :, 0]

    parsed_gvecs = np.split(gvecs, indices, axis=0)
    parsed_coeffs = np.split(complex_coeffs, indices, axis=1)
    return parsed_coeffs, parsed_gvecs
