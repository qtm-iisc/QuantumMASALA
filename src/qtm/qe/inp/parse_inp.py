import os
import warnings

from typing import Any, Union

from qtm.constants import RYDBERG, ANGSTROM
from qtm.crystal.crystal import Crystal
from .read_inp import PWscfIn
from .ibrav2latvec import *


from qtm.crystal import BasisAtoms
from qtm.lattice import RealLattice
from qtm.pseudo.upf import UPFv2Data
from qtm.kpts import KList
from qtm.kpts import gen_monkhorst_pack_grid

from qtm.pseudo.upf import _LIBXC_MAP

EPS5 = 1e-5


def raise_err(var: str, val: Any, allowed: Union[list[Any], str]):
    raise ValueError(
        f"value of '{var}' not valid / supported. Got {val}. "
        f"Allowed values: {allowed}"
    )


def warn(var: str, val: Any, default: Any):
    warnings.warn(
        f"value of '{var}' is not valid / supported. Got {val}"
        f"Replaced with: {default}"
    )


def gen_grid_shape(latvec: np.ndarray, ecutwfc: float):
    n1 = int(np.ceil(2 * np.sqrt(2 * ecutwfc) / np.pi * np.linalg.norm(latvec[:, 0])))
    n2 = int(np.ceil(2 * np.sqrt(2 * ecutwfc) / np.pi * np.linalg.norm(latvec[:, 1])))
    n3 = int(np.ceil(2 * np.sqrt(2 * ecutwfc) / np.pi * np.linalg.norm(latvec[:, 2])))

    good_primes = [2, 3, 5]

    n_new = []
    for ni in [n1, n2, n3]:
        is_good = False
        while not is_good:
            n = ni
            for prime in good_primes:
                while n % prime == 0:
                    n = n / prime
            if n == 1:
                is_good = True
            else:
                ni += 1
        n_new.append(ni)

    return tuple(n_new)


def parse_inp(pwin: PWscfIn):
    electrons = pwin.electrons

    # Validating Namelist 'CONTROL'
    allowed = ["scf", "nscf", "bands"]
    if pwin.control.calculation not in allowed:
        raise_err("calculation", pwin.control.calculation, allowed)

    allowed = ["high", "low"]
    if pwin.control.verbosity not in allowed:
        warn("verbosity", pwin.control.verbosity, "high")
        pwin.control.verbosity = "high"

    allowed = ["from_scratch", "restart"]
    if pwin.control.restart_mode not in allowed:
        raise_err("restart_mode", pwin.control.restart_mode, allowed)

    if pwin.control.iprint <= 0:
        warn("iprint", pwin.control.iprint, 100000)
        pwin.control.iprint = 100000

    if pwin.control.max_seconds <= 0:
        warn("max_seconds", pwin.control.max_seconds, 1e7)
        pwin.control.max_seconds = 1e7

    if not os.path.exists(pwin.control.pseudo_dir):
        raise ValueError(
            f"'pseudo_dir'={pwin.control.pseudo_dir} directory does not exit"
        )

    # Validating Namespace 'CONTROL'
    # Part 1: Constructing Lattice Parameters and generating `realspc` and `recispc`
    if pwin.system.ibrav == 0:
        print(pwin.system.celldm)
        option, cellparam = pwin.cell_parameters
        alat, latvec = cellparam2latvec(
            option, np.array(cellparam, dtype="f8"), pwin.system.celldm, pwin.system.a
        )
        pwin.system.celldm[1] = alat
    else:
        if pwin.system.ibrav not in ibrav_list:
            raise_err("ibrav", pwin.system.ibrav, list(ibrav_list.keys()))
        if pwin.system.celldm[1] is None:
            celldm = trad2celldm(
                pwin.system.ibrav,
                pwin.system.a,
                pwin.system.b,
                pwin.system.c,
                pwin.system.cosab,
                pwin.system.cosbc,
                pwin.system.cosac,
            )
            pwin = pwin._replace(system=pwin.system._replace(celldm=celldm))
        alat, latvec = ibrav2latvec(pwin.system.ibrav, pwin.system.celldm)
    for i in range(6):
        if i + 1 not in pwin.system.celldm:
            pwin.system.celldm[i + 1] = 0.0
    # realspc = RealSpace(alat, latvec)
    realspc = RealLattice(alat, latvec)

    if pwin.system.ecutwfc <= 0:
        raise_err("ecutwfc", pwin.system.ecutwfc, "positive float")

    if pwin.system.ecutrho is not None:
        warn("ecutrho", pwin.system.ecutrho, 4 * pwin.system.ecutwfc)
    pwin = pwin._replace(system=pwin.system._replace(ecutrho=4 * pwin.system.ecutwfc))

    if None in [pwin.system.nr1, pwin.system.nr2, pwin.system.nr3]:  # TODO: Warn?
        nr1, nr2, nr3 = gen_grid_shape(realspc.latvec, pwin.system.ecutwfc * RYDBERG)
        pwin = pwin._replace(system=pwin.system._replace(nr1=nr1, nr2=nr2, nr3=nr3))
    # Part 2: Generating `l_species` before validating the rest of parameters
    l_sp = {}
    for isp in range(pwin.system.ntyp):
        label, mass, upfdir = pwin.atomic_species[1][isp]
        upfdir = os.path.join(pwin.control.pseudo_dir, upfdir)
        if not os.path.isfile(upfdir):
            raise ValueError(f"UPF File '{upfdir}' missing for species '{label}'")
        try:
            ppdata = UPFv2Data.from_file(upfdir)
        except Exception as e:
            raise ValueError(
                f"failed to parse UPF File '{upfdir}' for species '{label}'"
            )

        l_sp[label] = [mass, ppdata, []]

    for iat in range(pwin.system.nat):
        label, coords = (
            pwin.atomic_positions[1][iat][0],
            pwin.atomic_positions[1][iat][1:],
        )
        if label not in l_sp:
            raise ValueError(f"species '{label}' not specified in 'ATOMIC_SPECIES'")
        l_sp[label][2].append(coords)

    l_species = []
    coords_typ = pwin.atomic_positions[0]
    for label, data in l_sp.items():
        mass, ppdata, coords = data
        if coords_typ == "crystal":
            l_pos_cryst = np.array(coords)
        elif coords_typ in ["alat", "bohr", "angstrom"]:
            if coords_typ == "alat":
                coords = pwin.system.celldm[1] * np.array(coords)
            elif coords_typ == "angstrom":
                coords = ANGSTROM * np.array(coords)
            l_pos_cryst = realspc.cart2cryst(coords, axis=1)
        else:
            raise ValueError(
                f"option '{coords_typ}' in card 'ATOMIC_POSITIONS' not valid / supported"
            )
        l_pos_cryst = l_pos_cryst.T
        # l_species.append(AtomicSpecies(label, mass, ppdata, realspc, l_pos_cryst))
        l_species.append(BasisAtoms(label, ppdata, mass, realspc, l_pos_cryst))

    l_xcfunc = [sp.ppdata.functional.lower() for sp in l_species]
    l_xcfunc = [_LIBXC_MAP[xc] for xc in l_xcfunc]
    input_dft = (
        _LIBXC_MAP[pwin.system.input_dft] if pwin.system.input_dft is not None else None
    )

    if input_dft is None:
        if not all(l_xcfunc[0] == xc for xc in l_xcfunc[1:]):
            raise ValueError(
                f"input pseudopotential do not have the same XC functional. Use 'input_dft' to override"
            )
        input_dft = l_xcfunc[0]
    else:
        for isp, xc in enumerate(l_xcfunc):
            warnings.warn(
                f"overriding xc functional specified in pseudopotential '{l_species[isp].ppdata.fname}'\n"
                f"'{l_species[isp].ppdata.functional}' -> {input_dft}"
            )

    pwin = pwin._replace(system=pwin.system._replace(input_dft=input_dft))
    cryst = Crystal(realspc, l_species)

    if pwin.system.noncolin:
        pwin.system.nspin = 2
        raise NotImplementedError("Noncolinear calculation yet to be implemented")

    numel = 0
    for sp in cryst.l_atoms:
        numel += sp.numatoms * sp.ppdata.z_valence

    if pwin.system.occupations not in ["fixed", "smearing"]:
        raise_err("occupations", pwin.system.occupations, ["fixed", "smearing"])

    if pwin.system.occupations == "fixed":
        if pwin.system.nspin == 2:
            raise NotImplementedError(
                "occupation='fixed' and nspin=2 requires setting total magnetization "
                "which is yet to be implemented"
            )
        if pwin.system.nspin == 1 and abs(numel % 2) > EPS5:
            raise ValueError(
                "number of electrons per unit cell must be even for 'fixed' occupations"
            )
        nbnd = int(numel // 2)
        pwin = pwin._replace(system=pwin.system._replace(nbnd=nbnd))
    else:
        if pwin.system.smearing in ["gaussian", "gauss"]:
            smearing = "gauss"
        elif pwin.system.smearing in ["methfessel-paxton", "m-p", "mp"]:
            smearing = "mp"
        elif pwin.system.smearing in ["marzari-vanderbilt", "cold", "m-v", "mv"]:
            smearing = "mv"
        elif pwin.system.smearing in ["fermi-dirac", "f-d", "fd"]:
            smearing = "fd"
        else:
            raise raise_err(
                "smearing",
                pwin.system.smearing,
                [
                    "gaussian",
                    "gauss",
                    "methfessel-paxton",
                    "m-p",
                    "mp",
                    "marzari-vanderbilt",
                    "cold",
                    "m-v",
                    "mv",
                    "fermi-dirac",
                    "f-d",
                    "fd",
                ],
            )
        pwin = pwin._replace(system=pwin.system._replace(smearing=smearing))

        if pwin.system.degauss <= EPS5:
            raise_err(
                "degauss", pwin.system.degauss, "any positive float larger than 10^-5"
            )

        if pwin.system.nbnd is not None:
            if 2 * pwin.system.nbnd < numel:
                raise ValueError(f"given 'nbnd' inadequate for calculation")
        else:
            nbnd = int(max(1.2 * (numel / 2), (numel / 2) + 4))
            pwin = pwin._replace(system=pwin.system._replace(nbnd=nbnd))

    if pwin.system.nspin == 2 and pwin.electrons.startingpot == "atomic":
        if len(pwin.system.starting_magnetization) == 0:
            raise ValueError(
                f"'starting_magnetization' need to be specified for atleast 1 element for LSDA"
            )
        if not all(
            isp - 1 in range(pwin.system.ntyp)
            for isp in pwin.system.starting_magnetization
        ):
            raise ValueError(f"'starting_magnetization' given for unspecified species")

        for isp in range(pwin.system.ntyp):
            if isp + 1 not in pwin.system.starting_magnetization:
                pwin.system.starting_magnetization[isp + 1] = 0.0
            if pwin.system.starting_magnetization[isp + 1] < -1.0:
                pwin.system.starting_magnetization[isp + 1] = -1.0
            elif pwin.system.starting_magnetization[isp + 1] > 1.0:
                pwin.system.starting_magnetization[isp + 1] = 1.0

    # elif pwin.electrons.startingpot == "atomic":
    #     warnings.warn(
    #         "'starting_magnetization' not required for nspin=1 and is ignored"
    #     )

    # Validating Namelist 'ELECTRONS'
    if electrons.electron_maxstep <= 0:
        raise_err(
            "electron_maxstep", electrons.electron_maxstep, "any positive integer"
        )

    if electrons.conv_thr <= 0:
        raise_err("conv_thr", electrons.conv_thr, "any positive float")

    allowed = ["plain"]
    if electrons.mixing_mode not in allowed:
        raise_err("mixing_mode", electrons.mixing_mode, allowed)

    if electrons.mixing_beta < 0 or electrons.mixing_beta > 1:
        raise_err(
            "mixing_beta", electrons.mixing_beta, "any positive float less than 1"
        )

    if electrons.mixing_ndim < 0:
        raise_err("mixing_ndim", electrons.mixing_ndim, "any positive integer")

    allowed = ["david"]
    if electrons.diagonalization not in allowed:
        raise_err("diagonalization", electrons.diagonalization, allowed)
    if electrons.diagonalization == "david":
        if electrons.diago_david_ndim < 0:
            raise_err(
                "diago_david_ndim", electrons.diago_david_ndim, "any positive integer"
            )

    if electrons.diago_thr_init <= 0:
        raise_err("diago_thr_init", electrons.diago_thr_init, "any positive float")

    allowed = ["atomic"]  # TODO: Update after implementing
    if electrons.startingpot not in allowed:
        raise_err("startingpot", electrons.startingpot, allowed)

    allowed = ["random"]
    if electrons.startingwfc not in allowed:
        raise_err("startingwfc", electrons.startingwfc, allowed)

    # Generate kpts_kgrp
    k_points = pwin.k_points
    option = k_points[0]

    allowed = ["tpiba", "automatic", "crystal"]
    if option not in allowed:
        raise ValueError(
            f"option of 'K_POINTS' card not valid / supported. Got {option}\n"
            f"Allowed values: {allowed}"
        )

    if option == "automatic":
        mpmesh_params = k_points[1][0].split()
        if len(mpmesh_params) != 6:
            raise ValueError(
                f"incomplete parameter list for generating k-points in 'automatic. Got {mpmesh_params}"
                "Expected: 3 positive integers (grid shape) followed by 3 booleans (shifts)"
            )
        try:
            grid_shape = tuple(int(n) for n in mpmesh_params[:3])
            shifts = tuple(bool(n) for n in mpmesh_params[3:])
        except ValueError as e:
            raise ValueError(
                f"invalid parameter list for generating k-points in 'automatic. Got {mpmesh_params}"
                "Expected: 3 positive integers (grid shape) followed by 3 booleans (shifts)"
            )
        kpts = gen_monkhorst_pack_grid(
            cryst, grid_shape, shifts, not pwin.system.nosym, not pwin.system.noinv
        )
    else:
        kpts = [
            np.fromiter(arr.split(), dtype="f8", count=4) for arr in pwin.k_points[1]
        ]
        kpts = np.array(kpts, dtype="f8")
        k_coords, k_weights = kpts[:, :3].T, kpts[:, 3]
        if option == "crystal":
            k_cryst = k_coords
        elif option == "tpiba":
            k_cryst = cryst.recispc.tpiba2cryst(k_coords)

        kpts = KList(cryst.recispc, k_cryst, k_weights, 'cryst')

    return pwin, cryst, kpts
