import os
import numpy as np
import h5py

from pypwscf.gspc import GSpace
from pypwscf.ele import ElectronDen, ElectronWfcBgrp


def read_chargeden(rho: ElectronDen, chden_dir: str):
    grho = rho.grho
    f = h5py.File(chden_dir, "r")

    for i in range(3):
        if not np.allclose(
            f["MillerIndices"].attrs[f"bg{i}"], grho.recispc.axes_cart[i]
        ):
            raise ValueError(
                f"'charge-density.hdf5' incompatibe. Reciprocal Axis 'bg{i+1}' mismatch"
            )

    if f.attrs["ngm"] != grho.numg:
        raise ValueError(
            f"'charge-densitt.hdf5' incompatible. Size of G-Space does not match. "
            f"Expected {grho.numg}, got{f['ngm']}"
        )

    miller_indices = np.array(f["MillerIndices"])
    idxsort = np.lexsort(
        (miller_indices[:, 2], miller_indices.T[:, 1], miller_indices[:, 0])
    )

    if not np.allclose(miller_indices.T[idxsort], grho.cryst):
        raise ValueError(
            f"'charge-density.hdf5' incompatible. G-Space vectors do not match"
        )

    numspin = f.attrs["nspin"]

    rho_g = np.empty((numspin, grho.numg), dtype="c16")

    rho_g[0] = np.array(f["rhotot_g"]).view("c16")[idxsort]

    if numspin == 2:
        rho_g[1] = np.array(f["rhodiff_f"]).view("c16")[idxsort]
        rho_g[0] = np.sum(rho_g, axis=0) / 2
        rho_g[1] = rho_g[0] - rho_g[1]

    f.close()
    rho_g *= np.prod(grho.grid_shape)
    rho.update_rho(rho_g)
    return rho


def write_chargeden(rho: ElectronDen, chden_dir: str):
    grho = rho.grho
    f = h5py.File(chden_dir, "w")

    numspin = rho.g.shape[0]
    f.attrs.create("gamma_only", np.array(b".FALSE.", dtype=np.bytes_))
    f.attrs.create("ngm_g", grho.numg, dtype="i4")
    f.attrs.create("nspin", numspin, dtype="i4")

    idxsort = np.lexsort(
        (
            grho.cryst[2],
            grho.cryst[1],
            grho.cryst[0],
            np.around(grho.norm2 / grho.recispc.tpiba**2, decimals=5),
        )
    )

    f.create_dataset("MillerIndices", grho.cryst.T, [idxsort], dtype="i4")
    for i in range(3):
        f["MillerIndices"].attrs.create(
            f"bg{i+1}", grho.recispc.axes_cart[i], dtype="f8"
        )

    rho_g = rho.g[(slice(None), idxsort)]
    rho_g /= np.prod(grho.grid_shape)

    rhotot_g = np.sum(rho_g, axis=0)
    f.create_dataset("rhotot_g", rhotot_g.view("f8"), dtype="f8")

    if numspin == 2:
        rhodiff_g = rho_g[0] - rho_g[1]
        f.create_dataset("rhodiff_g", rhodiff_g.view("f8"), dtype="f8")

    f.close()


def read_wfc(wfc: ElectronWfcBgrp, wfc_dir: str):
    gspc, gwfc = wfc.gspc, wfc.gwfc
    kpts = wfc.kpts
    numspin = wfc.numspin
    kpts_cart = kpts.cart.T

    # TODO: Code goes here for non-colinear

    for ik in range(kpts.numk):
        idxk_world = kpts.l_idxk[ik]
        for isp in range(numspin):
            if numspin == 1:
                fname = f"wfc{idxk_world+1}.hdf5"
            else:
                fname = f"wfc{'up' if numspin == 0 else 'dw'}{idxk_world+1}.hdf5"
            f = h5py.File(os.path.join(wfc_dir, fname), "r")

            for i in range(3):
                if not np.allclose(
                    f["MillerIndices"].attrs[f"bg{i}"], gspc.recispc.axes_cart[i]
                ):
                    raise ValueError(
                        f"'{fname}' incompatibe. Reciprocal Axis 'bg{i + 1}' mismatch"
                    )

            if not np.allclose(f.attrs["xk"], kpts_cart[ik]):
                raise ValueError(
                    f"'{fname}' incompatible. Mismatch in k-point coordinates"
                )

            if gwfc.l_numg_k[ik] != f.attrs["igwx"]:
                raise ValueError(
                    f"'{fname}' incompatible. Size of G-Space do not match"
                )

            miller_indices = np.array(f["MillerIndices"])
            idxsort = np.lexsort(
                (miller_indices[:, 2], miller_indices[:, 1], miller_indices[:, 0])
            )

            if not np.allclose(miller_indices[idxsort], gspc.cryst.T[gwfc.l_idxg(ik)]):
                raise ValueError(
                    f"'{fname}' incompatible. G-Space vectors do not match"
                )

            wfc.l_evc_gk[ik][isp][:] = np.array(f["evc"]).view("c16")[
                (wfc.idx, idxsort)
            ]
