template_dict = {
    "eps_header": {
        "flavor": "int32",
        "freqs": {
            "freq_dep": "int32",
            "freqs": "<f8",
            "nfreq": "int32",
            "nfreq_imag": "int32",
        },
        "gspace": {
            "ekin": "<f8",
            "gind_eps2rho": "int32",
            "gind_rho2eps": "int32",
            "nmtx": "int32",
            "nmtx_max": "int32",
            "vcoul": "<f8",
        },
        "params": {
            "ecuts": "<f8",
            "efermi": "<f8",
            "has_advanced": "int32",
            "icutv": "int32",
            "intraband_flag": "int32",
            "intraband_overlap_min": "<f8",
            "matrix_flavor": "int32",
            "matrix_type": "int32",
            "nband": "int32",
            "nmatrix": "int32",
            "subsample": "int32",
            "subspace": "int32",
        },
        "qpoints": {
            "nq": "int32",
            "qgrid": "int32",
            "qpt_done": "int32",
            "qpts": "<f8",
        },
        "versionnumber": "int32",
    },
    # 'mats': {
    # 'matrix': '<f8',
    # 'matrix-diagonal': '<f8'},
    # 'mf_header': {
    #    'crystal': {
    #        'adot': '<f8',
    #         'alat': '<f8',
    #         'apos': '<f8',
    #         'atyp': 'int32',
    #         'avec': '<f8',
    #         'bdot': '<f8',
    #         'blat': '<f8',
    #         'bvec': '<f8',
    #         'celvol': '<f8',
    #         'nat': 'int32',
    #         'recvol': '<f8'},
    #     'flavor': 'int32',
    #     'gspace': {
    #         'FFTgrid': 'int32',
    #         'components': 'int32',
    #         'ecutrho': '<f8',
    #         'ng': 'int32'},
    #     'kpoints': {
    #         'ecutwfn': '<f8',
    #         'el': '<f8',
    #         'ifmax': 'int32',
    #         'ifmin': 'int32',
    #         'kgrid': 'int32',
    #         'mnband': 'int32',
    #         'ngk': 'int32',
    #         'ngkmax': 'int32',
    #         'nrk': 'int32',
    #         'nspin': 'int32',
    #         'nspinor': 'int32',
    #         'occ': '<f8',
    #         'rk': '<f8',
    #         'shift': '<f8',
    #         'w': '<f8'},
    #     'symmetry': {
    #         'cell_symmetry': 'int32',
    #         'mtrx': 'int32',
    #         'ntran': 'int32',
    #         'tnp': '<f8'},
    #         'versionnumber': 'int32'}
}


import h5py
import numpy as np

from qtm.interfaces.bgw.h5_utils import (
    cplx_to_real,
    create_empty_h5,
    real_to_cplx,
    write_recursive_h5,
)

TOLERANCE = 1e-5


def unpad_array(arr):
    # Remove padded rows
    for i in range(arr.shape[-2] - 1, -1, -1):
        if np.allclose(arr[..., i, :], 0):
            arr = arr[..., :i, :]
        else:
            break

    # Remove padded cols
    for j in range(arr.shape[-1] - 1, -1, -1):
        if np.allclose(arr[..., j], 0):
            arr = arr[..., :j]
        else:
            break

    # arr should be square
    mindim = min(arr.shape[-2:])
    # print(mindim)

    return arr[..., :mindim, :mindim]


def write_mats(filename, mats, auxfile=None):
    """Write matrix to h5 file
    Expects a list of epsilon matrices, one for each q-point.
    Expected type: List[ndarray[n,n]]"""
    f = h5py.File(filename, "w")

    # Create empty h5 group with correct dtypes
    create_empty_h5(f, template_dict)

    # Copy some entries from source
    if auxfile:
        aux = h5py.File(auxfile, "r")
        f.create_group("mf_header")
        write_recursive_h5(f["mf_header"], aux["mf_header"], sourcetype=h5py.Group)
        aux.close()

    # Find the maximum size of matrices in mats
    # in order to be able to create a single 'matrix' array
    max_dim = max([max(mat.shape) for mat in mats])

    # Pad matrices with zeroes to increase size to the ommon maximum
    # Exactly same thing has been done in BGW
    # Pad square matrices with zeroes to increase size
    np_mats = cplx_to_real(
        np.stack(
            [
                np.pad(
                    mat,
                    pad_width=(
                        (0, max_dim - mat.shape[0]),
                        (0, max_dim - mat.shape[1]),
                    ),
                    mode="constant",
                    constant_values=(0.0, 0.0),
                )
                for mat in mats
            ]
        )
    )

    # Populate known entities
    f.create_group("mats")
    f["mats"].create_dataset("matrix", data=np_mats)

    f["mats"].create_dataset(
        "matrix-diagonal",
        data=np.stack([np.diagonal(mat.T, axis1=1, axis2=2) for mat in np_mats]),
    )

    f.close()


def read_mats(filename):
    """Read matrix stroed at `mats/matrix` from h5 file.
    Intended to be used to read epsmats and chimats.
    Converts Real to Complex and unpads the data (reuired for matrices that have smaller dimension than others, but were padded with zeroes to pt them all in a single array.).
    """
    f = h5py.File(filename, "r")

    matrix = np.array(f["mats"]["matrix"])
    matrix = np.conj(real_to_cplx(matrix))
    mats = []
    for mat in matrix:
        mats.append(unpad_array(mat))
    f.close()
    return mats


if __name__ == "__main__":
    mat = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=float)
    # debug("mat")

    # debug("mat.shape")
    # debug("unpad_array(mat).shape")
    # debug("unpad_array(mat)")

    # mat = [np.arange(24).reshape(6,4),np.arange(5*7).reshape(5,7)]
    # write_mats("testmat.h5", mat)#, "wfn_cplx.h5")

    # print(*mat, sep="\n")

    # mats = read_mats("testmat.h5")
    # print(*mats, sep="\n")
