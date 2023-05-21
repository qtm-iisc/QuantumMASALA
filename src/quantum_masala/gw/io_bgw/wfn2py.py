from collections import namedtuple
from typing import List, NamedTuple
import numpy as np
import h5py
from quantum_masala.core import RealLattice, AtomBasis, Crystal, GSpace, KList
from quantum_masala.core import GkSpace  # , WfnK, Wavefun
from quantum_masala.dft.kswfn import KSWavefun
from quantum_masala.constants import RYDBERG


class Symmetry(NamedTuple):
    ntran: int
    cell_symmetry: int
    mtrx: np.ndarray
    tnp: np.ndarray

class SymmetrySubgroup(NamedTuple):
    ntran: int
    cell_symmetry: int
    mtrx: np.ndarray
    tnp: np.ndarray
    indsub: List[int]
    kgzero: List[np.ndarray]
    qvec: np.ndarray
    supergroup: Symmetry




class WfnData(NamedTuple):
    crystal: Crystal
    grho: GSpace
    kpts: KList
    l_gk: List[GkSpace]
    l_wfn: List[KSWavefun]
    symmetry: Symmetry

class UnfoldedWfnData(NamedTuple):
    wfndata: WfnData
    crystal: Crystal
    grho: GSpace
    kpts: KList
    l_gk: List[GkSpace]
    l_wfn: List[KSWavefun]


def wfn2py(filename="../test/bgw/WFN.h5", verbose=False):
    """
    Converts WFN[q].h5 to a set of python objects from qtm.core library.

    Parameters
    ----------
    filename : str
        Address of WFN[q].h5 file.

    Returns
    -------
    crystal : Crystal
    grho : GSpace
    kpts : KList
    l_gk: List[GkSpace]
    l_wfn : list[KSWavefun]
    """

    # ---- Load wfn.h5 data ---------------------------------------------------
    # if filename_q == None:
    #     filename_q = filename[:-3] + "q.h5"
    # Take care of it by calling wfn2py separately again.

    if verbose:
        print("WFN file : ", filename)

    wfn_ = h5py.File(filename, "r")
    if verbose:
        print(wfn_.keys())

    wfns_ = wfn_["wfns"]
    mf_header = wfn_["mf_header"]
    kpoints = mf_header["kpoints"]
    gspace = mf_header["gspace"]
    symmetry = mf_header["symmetry"]
    crystal_ = mf_header["crystal"]

    if verbose:
        print(
            wfns_["coeffs"].shape
        )  # Shape does not match the 'WFN.h5' docs. Shape is transposed

    # Crystal
    alat = np.array(crystal_["alat"])
    nat = np.array(crystal_["nat"])
    avec = np.array(crystal_["avec"])
    atyp = np.array(crystal_["atyp"])
    apos = np.array(crystal_["apos"])
    if verbose:
        print("alat, nat, avec, atyp, apos")
        print(alat)
        print(nat)
        print(avec)
        print(atyp)
        print(apos)

    # Generating 'reallat'
    reallat = RealLattice.from_alat(alat, *avec)

    # Parsing 'atyp' and 'apos' to generate 'l_species'
    isort = np.argsort(atyp)
    l_typ, l_counts = np.unique(atyp, return_counts=True)
    l_coords = np.split(apos[isort], np.cumsum(l_counts))
    
    l_species = []
    mnband = np.array(kpoints["mnband"])
    occ = np.array(kpoints["occ"])
    valence = np.sum(occ) / kpoints["nrk"]

    for ityp, typ_num in enumerate(l_typ):
        l_species.append(
            AtomBasis.from_alat(
                typ_num, None, None, reallat, *l_coords[ityp], valence=valence
            )
        )

    # Finally creating 'Crystal' instance
    crystal = Crystal(reallat, l_species)

    reallat, recilat, l_species = crystal.reallat, crystal.recilat, crystal.l_atoms

    assert np.allclose(reallat.axes_alat, np.array(crystal_["avec"])), "avec"
    assert np.allclose(reallat.alat, np.array(crystal_["alat"])), "alat"
    assert np.allclose(reallat.cellvol, np.array(crystal_["celvol"])), "celvol"
    assert np.allclose(reallat.adot, np.array(crystal_["adot"])), "adot"
    assert np.allclose(recilat.tpiba, np.array(crystal_["blat"])), "blat"
    assert np.allclose(recilat.axes_tpiba, np.array(crystal_["bvec"])), "bvec"
    assert np.allclose(recilat.cellvol, np.array(crystal_["recvol"])), "recvol"
    assert np.allclose(recilat.bdot, np.array(crystal_["bdot"])), "bdot"

    if verbose:
        print("Bravais Lattice")
        print(f"Lattice parameter 'a': alat = {reallat.alat} bohr")
        print("Axes of bravais lattice in units of alat:")
        for i, ax in enumerate(reallat.axes_alat):
            print(f"a{i+1} = {ax}")

        print("Checking additional values in bgw's 'wfn.h5'")
        print("alat: ", np.allclose(reallat.alat, np.array(crystal_["alat"])))
        print("avec: ", np.allclose(reallat.axes_alat, np.array(crystal_["avec"])))
        print("celvol: ", np.allclose(reallat.cellvol, np.array(crystal_["celvol"])))
        print("adot: ", np.allclose(reallat.adot, np.array(crystal_["adot"])))
        print("-" * 40)
        print()

        print("Reciprocal Lattice")
        print(f"Lattice parameter 'b': blat = {recilat.tpiba} bohr^-1")
        print("Axes of reciprocal lattice in units of tpiba:")
        for i, bx in enumerate(recilat.axes_tpiba):
            print(f"b{i+1} = {bx}")

        print("Checking additional values in bgw's 'wfn.h5'")
        print("blat: ", np.allclose(recilat.tpiba, np.array(crystal_["blat"])))
        print("bvec: ", np.allclose(recilat.axes_tpiba, np.array(crystal_["bvec"])))
        print("recvol: ", np.allclose(recilat.cellvol, np.array(crystal_["recvol"])))
        print("bdot: ", np.allclose(recilat.bdot, np.array(crystal_["bdot"])))
        print("-" * 40)
        print()

        print("Atoms in Unit Cell")
        for isp, sp in enumerate(l_species):
            print(f"Species #{isp+1}: Atomic Number - {sp.label}")
            print(f"Coordinates in units of alat")
            for iat, r_alat in enumerate(sp.alat.T):  # Note the transpose
                print(r_alat)
        print("-" * 40)
        print()

    # # GSpace
    # ========
    ecutrho = np.array(gspace["ecutrho"])
    FFTgrid = np.array(gspace["FFTgrid"])

    grho = GSpace(crystal, ecutrho * RYDBERG, tuple(FFTgrid))
    # NOTE: Check if FFTgrid needs to be reversed by generating WFN.h5
    # for a unequal mesh via QE input 'nr1, nr2, nr3' arguments

    assert grho.numg == np.array(
        gspace["ng"]
    ), "Mismatch: Number of vectors in G-Sphere"

    components = np.array(gspace["components"])
    isort_bgw = np.lexsort([components[:, 2], components[:, 1], components[:, 0]])
    isort_py = np.lexsort([grho.cryst[2], grho.cryst[1], grho.cryst[0]])

    assert np.allclose(
        components[isort_bgw], grho.cryst.T[isort_py]
    ), "Mismatch: G-space components"

    # Generating a mapping between 'grho.cryst' and 'gspace["components"]'
    idx_grho_bgw2pw = np.empty(grho.numg, dtype="i4")
    idx_grho_bgw2pw[isort_py] = np.arange(grho.numg, dtype="i4")[isort_bgw]

    assert np.allclose(
        grho.cryst.T, components[idx_grho_bgw2pw]
    ), "Failed testing generated mapping from bgw to pw"

    #  KList
    # ==========
    w = np.array(kpoints["w"])
    rk = np.array(kpoints["rk"])  # determined to be in cryst

    kpts = KList.from_cryst(crystal, *zip(rk, w))

    assert np.allclose(kpts.numk, np.array(kpoints["nrk"])), "Mismatch: nrk & numk"
    assert np.allclose(
        kpts.cryst, np.array(kpoints["rk"])
    ), "Mismatch: kgrid components rk & cryst.T"
    assert np.allclose(kpts.weights, np.array(kpoints["w"])), "Mismatch: kgrid weights"

    if verbose:
        print("Checking KList instance")
        print("Number: ", np.allclose(kpts.numk, np.array(kpoints["nrk"])))
        print("Coordinates: ", np.allclose(kpts.cryst.T, np.array(kpoints["rk"])))
        print("Weights: ", np.allclose(kpts.weights, np.array(kpoints["w"])))

    # GkSpace
    # =========
    ecutwfc = np.array(kpoints["ecutwfc"])

    wfn_gspc = GSpace(
        crystal=crystal, ecut=4 * ecutwfc * RYDBERG, grid_shape=grho.grid_shape
    )

    l_gk = [GkSpace(wfn_gspc, k_cryst) for k_cryst in kpts.cryst]

    assert np.allclose(
        [gk.numgk for gk in l_gk], np.array(kpoints["ngk"])
    ), "Mismatch: ngk & [gk.numgk]"

    gvecs = np.array(wfns_["gvecs"])
    l_gk_bgw = np.split(
        np.array(wfns_["gvecs"]), np.cumsum(np.array(kpoints["ngk"])[:-1])
    )

    for i, gk in enumerate(l_gk):
        gcryst_py = gk.g_cryst
        gcryst_bgw = l_gk_bgw[i]
        isort_bgw = np.lexsort([gcryst_bgw[:, 2], gcryst_bgw[:, 1], gcryst_bgw[:, 0]])
        isort_py = np.lexsort([gcryst_py[2], gcryst_py[1], gcryst_py[0]])
        idxbgw2pw = np.empty(gk.numgk, dtype="i4")
        idxbgw2pw[isort_py] = np.arange(gk.numgk, dtype="i4")[isort_bgw]
        gk.idxbgw2pw = idxbgw2pw

        assert np.allclose(gcryst_bgw[isort_bgw], gcryst_py.T[isort_py])
        assert np.allclose(gk.g_cryst.T, gcryst_bgw[gk.idxbgw2pw])
        if verbose:
            print(f"Validating 'gk' for k-point #{i+1}")
            print(
                "Components match: ",
                np.allclose(gcryst_bgw[isort_bgw], gcryst_py.T[isort_py]),
            )
            print(
                "Generated mapping is correct: ",
                np.allclose(gk.g_cryst.T, gcryst_bgw[gk.idxbgw2pw]),
            )
            print()

    # # Loading Wavefunctions
    # =======================
    nspin = np.array(kpoints["nspin"])
    nspinor = np.array(kpoints["nspinor"])  # Raise error when it is not 1
    mnband = np.array(kpoints["mnband"])
    el = np.array(kpoints["el"])
    occ = np.array(kpoints["occ"])
    is_spin = True if np.array(kpoints["nspin"])>1 else False

    # WfnK = Wavefun(wfn_gspc, int(nspin), int(mnband))

    l_wfn = [
        KSWavefun(gspc=wfn_gspc, k_cryst=k_cryst, k_weight=k_weight, numbnd=int(mnband), is_spin=is_spin, is_noncolin=False)
        for idxk, (k_cryst, k_weight) in enumerate(kpts)
    ]

    # wfn_generate(gspc=gspace,
    #              kpts=kpoints,
    #              numbnd=mnband[0],
    #              is_noncolin=False,
    #              is_spin=False)

    gvecs = np.array(wfn_["wfns/gvecs"])
    l_gk_bgw = np.split(
        np.array(wfn_["wfns/gvecs"]), np.cumsum(np.array(kpoints["ngk"])[:-1])
    )
    # print(wfns_["coeffs"].shape)
    l_coeffs = np.split(
        1j
        * np.conj(np.array(wfns_["coeffs"]).view("c16"))
        .reshape((mnband, nspin, -1))
        .transpose((1, 0, 2)),
        np.cumsum(np.array(kpoints["ngk"][:-1])),
        axis=2,
    )

    for ik, wfn in enumerate(l_wfn):
        wfn.evl[:] = kpoints["el"][:, ik, :]
        wfn.occ[:] = kpoints["occ"][:, ik, :]

        # Duplicated code for reference
        gk = wfn.gkspc  # Yet to refactor the RHS name from gwfc -> gk
        gcryst_py = gk.g_cryst
        gcryst_bgw = l_gk_bgw[ik]
        isort_bgw = np.lexsort([gcryst_bgw[:, 2], gcryst_bgw[:, 1], gcryst_bgw[:, 0]])
        isort_py = np.lexsort([gcryst_py[2], gcryst_py[1], gcryst_py[0]])
        idxbgw2pw = np.empty(gk.numgk, dtype="i4")
        idxbgw2pw[isort_py] = np.arange(gk.numgk, dtype="i4")[isort_bgw]
        gk.idxbgw2pw = idxbgw2pw
        # End of Duplicate code

        wfn.evc_gk[:] = l_coeffs[ik][:, :, gk.idxbgw2pw]
        # print(l_coeffs.shape)

        assert np.allclose(
            gcryst_bgw[isort_bgw], gcryst_py.T[isort_py]
        ), f"Mismatch: gvecs[ik={ik}]"
        assert np.allclose(
            gk.g_cryst.T, gcryst_bgw[gk.idxbgw2pw]
        ), f"Mismatch: gvecs[ik={ik}][gk.idxbgw2pw]"
        if verbose:
            print(
                f"Components match for ik={ik}: ",
                np.allclose(gcryst_bgw[isort_bgw], gcryst_py.T[isort_py]),
            )
            print(
                "Generated mapping is correct: ",
                np.allclose(gk.g_cryst.T, gcryst_bgw[gk.idxbgw2pw]),
            )
            print()

    # Symmetry
    symmetry = Symmetry(cell_symmetry=np.array(symmetry["cell_symmetry"]),
                        mtrx = np.array(symmetry["mtrx"]),
                        ntran = np.array(symmetry["ntran"]),
                        tnp = np.array(symmetry["tnp"]))

    if verbose:
        print("Loaded WFC Data")

    # Use *wfn2py(...) to unwrap the data in namedtuple, if needed
    # namedtuple makes it easier to call wfn2 py because the returned elements need not be remembered.
    return WfnData(crystal, grho, kpts, l_gk, l_wfn, symmetry)


if __name__ == "__main__":
    verbose = True
    nt = wfn2py("../test/bgw/WFN.h5", verbose)

    crystal, grho, kpts, l_gk, l_wfn = nt
    print(crystal, grho, kpts, l_gk, l_wfn, sep="\n\n")
    # print(kpts.cryst)
