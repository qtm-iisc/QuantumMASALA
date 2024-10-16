# ----------------------------------------------------------
# COLLECTION OF FUNCTIONS FOR HANDLING HDF5 READING/WRITING
#
# CONVERSION BETWEEN REAL AND COMPLEX ARRAYS
# * cplx_to_real(arr)
# * real_to_cplx(arr)
#
# CONVERSION BETWEEN HDF5 AND ARRAY
# * h5_to_arr
# * arr_to_h5
# * create_empty_h5
#
# FULL CONVERSION BETWEEN HDF5 AND DICTIONARY
# * dict_to_h5
# * read_input_h5
#
# -----------------------------------------------------------

from typing import Dict
import numpy as np
import h5py


# CONVERSION BETWEEN REAL AND COMPLEX ARRAYS ------------


def cplx_to_real(arr):
    """Convert complex array to real array
    with last dimention split into two 'columns'.
    First column real, second imaginary"""
    return np.stack([arr.real, arr.imag], axis=-1)


def real_to_cplx(arr):
    """Convert real array to complex array
    assuming last dimention contains
    first column imaginary, second real"""
    return arr[Ellipsis, 0] + 1j * arr[Ellipsis, 1]


# CONVERSION BETWEEN HDF5 AND ARRAY ---------------------


def h5_to_arr(filename, cplx=True, addr=None):
    f = h5py.File(filename, "r")
    if not addr:
        arr = np.array(f[list(f.keys())[0]])
        if cplx:
            arr = real_to_cplx(arr)
    else:
        arr = np.array(f[addr])
    return arr


def arr_to_h5(arr, filename, name="array", cplx=True):
    """Create file and write a single array"""
    f = h5py.File(filename, "w")
    if cplx:
        f.create_dataset(name, data=cplx_to_real(arr))
    else:
        f.create_dataset(name, data=arr)
    f.close()


# FULL CONVERSION BETWEEN HDF5 AND DICTIONARY ------------


def dict_to_h5(d, filename, name="data"):
    """Write dictionary to h5 file"""
    f = h5py.File(filename, "w")
    write_recursive_h5(f, d)
    f.close()


def write_recursive_h5(h: h5py.Group, d: Dict, sourcetype=dict):
    """Write dictionary d to header h"""
    for key in d:
        if isinstance(d[key], sourcetype):
            h.create_group(key)
            write_recursive_h5(h[key], d[key])
        else:
            h.create_dataset(key, data=d[key])
    return


def create_empty_h5(h: h5py.Group, d: dict):
    """Write template dictionary d to header h
    template dictionary contains hrpy datatypes as values
    and this function creates empty h5 datasets with those datatypes."""
    for key in d:
        if isinstance(d[key], dict):
            h.create_group(key)
            create_empty_h5(h[key], d[key])
        else:
            h.create_dataset(key, data=h5py.Empty(dtype=d[key]))
    return


# Reading WFN.h5 file to dict, with appropriate conversion and preparation


def read_input_h5(filename, wfn_prepare=False):
    """Read wfn.h5 to dict"""
    header = h5py.File(filename, "r")
    data = retrieve_dict(header)
    if wfn_prepare:
        prepare_data(data)  # Makes it specific to wfn.h5 and epsilon.py
    return data


def retrieve_dict(source):
    """A recursive function to read sub-groups as nested dictionaries.
    Datasets are converted to np.array ."""
    target = {}
    for sub in list(source.keys()):
        if isinstance(source[sub], h5py.Dataset):
            target[sub] = np.array(source[sub])
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
    complex_coeffs = coeffs[:, 0, :, 1] + 1j * coeffs[:, 0, :, 0]

    parsed_gvecs = np.split(gvecs, indices, axis=0)
    parsed_coeffs = np.split(complex_coeffs, indices, axis=1)
    return parsed_coeffs, parsed_gvecs
