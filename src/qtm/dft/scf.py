from __future__ import annotations
from typing import TYPE_CHECKING

from qtm.logger import qtmlogger
from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from qtm.mpi.containers import get_DistFieldG
from qtm.mpi.gspace import DistGSpace, DistGkSpace

if TYPE_CHECKING:
    from typing import Literal
    from numbers import Number
__all__ = ["scf", "EnergyData", "IterPrinter"]

from dataclasses import dataclass
from time import perf_counter
from sys import version_info
import numpy as np

from qtm.crystal import Crystal
from qtm.kpts import KList
from qtm.gspace import GSpace, GkSpace
from qtm.containers import FieldGType, FieldRType, get_FieldG
from qtm.dft.exchange import Exchange, MultiKExchangeDriver

from qtm.pot import hartree, xc, ewald
from qtm.pseudo import loc_generate_rhoatomic, loc_generate_pot_rhocore, NonlocGenerator
from qtm.symm.symmetrize_field import SymmFieldMod

from qtm.dft import DFTCommMod, DFTConfig, KSWfn, KSHam, eigsolve, occup, mixing

from qtm.mpi.check_args import check_system
from qtm.mpi.utils import scatter_slice

from qtm.msg_format import *
from qtm.constants import RYDBERG


@dataclass
class EnergyData:
    total: float = 0.0
    hwf: float = 0.0
    one_el: float = 0.0
    ewald: float = 0.0
    hartree: float = 0.0
    xc: float = 0.0
    exx: float = 0.0

    fermi: float | None = None
    smear: float | None = None
    internal: float | None = None

    HO_level: float | None = None
    LU_level: float | None = None


from qtm.io_utils.dft_printers import print_scf_parameters

if version_info[1] >= 8:
    from typing import Protocol

    class IterPrinter(Protocol):
        def __call__(
            self,
            idxiter: int,
            runtime: float,
            scf_converged: bool,
            e_error: float,
            diago_thr: float,
            diago_avgiter: float,
            en: EnergyData,
        ) -> None: ...

    class WfnInit(Protocol):
        def __call__(self, ik: int, kswfn: list[KSWfn]) -> None: ...

else:
    IterPrinter = "IterPrinter"
    WfnInit = "WfnInit"


@qtmlogger.time("scf:scf")
def scf(
    dftcomm: DFTCommMod,
    crystal: Crystal,
    kpts: KList,
    grho: GSpace,
    gwfn: GSpace,
    numbnd: int,
    is_spin: bool,
    is_noncolin: bool,
    symm_rho: bool = True,
    rho_start: FieldGType | tuple[float, ...] | None = None,
    wfn_init: WfnInit | None = None,
    libxc_func: tuple[str, str] | None = None,
    occ_typ: Literal["fixed", "smear"] = "smear",
    smear_typ: Literal["gauss", "fd", "mv"] = "gauss",
    e_temp: float = 1e-3,
    conv_thr: float = 1e-6 * RYDBERG,
    maxiter: int = 100,
    diago_thr_init: float = 1e-5 * RYDBERG,
    iter_printer: IterPrinter | None = None,
    mix_beta: float = 0.7,
    mix_dim: int = 8,
    dftconfig: DFTConfig | None = None,
    ret_vxc: bool = False,
    hf: bool = False,
    exx_maxstep: int = 12,
    adaptive_thr: bool = True,
    conv_thr_init: float = 1e-3 * RYDBERG,
    conv_thr_multi: float = 1e-1,
    exx_conv_thr: float = 1e-8 * RYDBERG,
    reset_mixing_on_exx: bool = True,
    exxdiv_treatment: str = "vcut_spherical",
    exx_Rc: float | None = None,
) -> (
    tuple[bool, FieldGType, list[list[KSWfn]], EnergyData, np.ndarray]
    | tuple[bool, FieldGType, list[list[KSWfn]], EnergyData]
):
    if not isinstance(dftcomm, DFTCommMod):
        raise TypeError(type_mismatch_msg("dftcomm", dftcomm, DFTCommMod))

    # Checking input arguments
    with dftcomm.image_comm as comm:
        check_system(comm, crystal, grho, gwfn, kpts)
        if grho is not gwfn:
            raise NotImplementedError()

        hf = comm.bcast(hf)
        assert isinstance(hf, bool)

        exx_maxstep = comm.bcast(exx_maxstep)
        assert isinstance(exx_maxstep, int)
        assert exx_maxstep > 0

        adaptive_thr = comm.bcast(adaptive_thr)
        assert isinstance(adaptive_thr, bool)

        conv_thr_init = comm.bcast(conv_thr_init)
        assert isinstance(conv_thr_init, float)
        assert conv_thr_init > 0

        conv_thr_multi = comm.bcast(conv_thr_multi)
        assert isinstance(conv_thr_multi, float)
        assert conv_thr_multi >= 0

        exx_conv_thr = comm.bcast(exx_conv_thr)
        assert isinstance(exx_conv_thr, float)
        assert exx_conv_thr > 0

        reset_mixing_on_exx = comm.bcast(reset_mixing_on_exx)
        assert isinstance(reset_mixing_on_exx, bool)

        exxdiv_treatment = comm.bcast(exxdiv_treatment)
        assert isinstance(exxdiv_treatment, str)

        exx_Rc = comm.bcast(exx_Rc)
        assert (exx_Rc is None) or isinstance(exx_Rc, float)
        if isinstance(exx_Rc, float):
            assert exx_Rc > 0.0

        numbnd = comm.bcast(numbnd)
        assert isinstance(numbnd, int)
        assert numbnd > 0

        is_spin = comm.bcast(is_spin)
        assert isinstance(is_spin, bool)
        is_noncolin = comm.bcast(is_noncolin)
        assert isinstance(is_noncolin, bool)
        if is_noncolin:
            raise NotImplementedError("noncollinear calculations yet to be implemented")
        if is_noncolin:
            is_spin = True
        if isinstance(gwfn, DistGSpace):
            is_gwfn_dist = True
        else:
            is_gwfn_dist = False

        symm_rho = comm.bcast(symm_rho)
        assert isinstance(symm_rho, bool)

        if isinstance(rho_start, FieldGType):
            assert rho_start.gspc is grho
            if rho_start.gspc is not grho:
                raise ValueError(
                    obj_mismatch_msg("rho_start.gspc", rho_start.gspc, "grho", grho)
                )
            if rho_start.shape != (1 + is_spin,):
                raise ValueError(
                    value_mismatch_msg(
                        "rho_start.shape",
                        rho_start.shape,
                        f"(1 + is_spin, ) = {(1 + is_spin, )}",
                    )
                )
        elif is_spin:
            if rho_start is None:
                raise ValueError(
                    type_mismatch_msg(
                        "rho_start",
                        rho_start,
                        f"a {get_FieldG} instance or a sequeuce of numbers between -1 and +1 "
                        f"representing starting spin polarisation on each atomic type of"
                        f"crystal (for is_spin = {is_spin})",
                    )
                )
            starting_mag = rho_start
            rho_start = get_FieldG(grho).zeros(1 + is_spin)
            for sp, mag in zip(crystal.l_atoms, starting_mag):
                rho_atomic_sp = loc_generate_rhoatomic(sp, grho)
                mag = max(min(mag, 1), -1)
                rho_start[0] += ((1 + mag) / 2) * rho_atomic_sp
                rho_start[1] += ((1 - mag) / 2) * rho_atomic_sp
        else:
            rho_start = sum(
                loc_generate_rhoatomic(sp, grho) for sp in crystal.l_atoms
            ).reshape(1)

        libxc_func = comm.bcast(libxc_func)

        occ_typ = comm.bcast(occ_typ)
        assert occ_typ in ["fixed", "smear"]

        smear_typ = comm.bcast(smear_typ)
        if occ_typ == "smear":
            assert smear_typ in ["gauss", "fd", "mv"]

        e_temp = comm.bcast(e_temp)
        assert isinstance(e_temp, float)
        assert e_temp >= 0

        conv_thr = comm.bcast(conv_thr)
        assert isinstance(conv_thr, float)
        assert conv_thr > 0

        maxiter = comm.bcast(maxiter)
        assert isinstance(maxiter, int)
        assert maxiter >= 0

        diago_thr_init = comm.bcast(diago_thr_init)
        assert isinstance(diago_thr_init, float)
        assert diago_thr_init > 0

        mix_beta = comm.bcast(mix_beta)
        assert isinstance(mix_beta, float)
        assert 0 < mix_beta <= 1

        mix_dim = comm.bcast(mix_dim)
        assert isinstance(mix_dim, int)
        assert mix_dim > 0

        if dftconfig is None:
            dftconfig = DFTConfig()
        comm.bcast(dftconfig)
        assert isinstance(dftconfig, DFTConfig)

    # HF/EXX with a k-point mesh supports k-point parallelism across
    # MPI k-groups. Each k-group must currently contain one MPI rank,
    # and distributed plane-wave GSpace is not yet supported.
    if hf and kpts.numkpts != 1:
        if isinstance(gwfn, DistGSpace):
            raise NotImplementedError(
                "HF/EXX with k-point meshes is not implemented for DistGSpace yet."
            )

        if dftcomm.kgrp_intra.size != 1:
            raise NotImplementedError(
                "HF/EXX k-point MPI currently supports one MPI rank per "
                "k-group only."
            )

    start_time = perf_counter()
    image_comm = dftcomm.image_comm
    pwgrp_inter_kgrp = dftcomm.pwgrp_inter_kgrp
    kroot_intra = dftcomm.kroot_intra

    assert (
        kpts.numkpts >= dftcomm.n_kgrp
    ), "Number of k-points must be greater than or equal to the number of k-groups"

    if dftcomm.image_comm.rank == 0:
        print_scf_parameters(
            dftcomm,
            crystal,
            grho,
            gwfn,
            numbnd,
            is_spin,
            is_noncolin,
            symm_rho,
            rho_start,
            wfn_init,
            libxc_func,
            occ_typ,
            smear_typ,
            e_temp,
            conv_thr,
            maxiter,
            diago_thr_init,
            iter_printer,
            mix_beta,
            mix_dim,
            dftconfig,
            ret_vxc,
            kpts,
        )

    with image_comm:
        n_kgrp, i_kgrp = dftcomm.n_kgrp, dftcomm.i_kgrp
        n_bgrp, i_bgrp = dftcomm.n_bgrp, dftcomm.i_bgrp
        i_kpts_kgrp = range(kpts.numkpts)[scatter_slice(kpts.numkpts, n_kgrp, i_kgrp)]

        symm_mod = None
        # if symm_rho:
        symm_mod = SymmFieldMod(crystal, grho)

        if wfn_init is None:

            def wfn_init(ik, kswfn_k):
                np.random.seed(ik)  # For reproducible speed measurements
                kswfn_k[0].init_random()
                if is_spin and not is_noncolin:
                    kswfn_k[1].init_random()

        l_kswfn_kgrp = []
        l_ksham_kgrp = []

        for ik in i_kpts_kgrp:
            k_cryst, k_weight = kpts[ik]

            if is_gwfn_dist:
                gkspc = GkSpace(gwfn.gspc_glob, k_cryst)
                gkspc = DistGkSpace(gwfn.pwgrp_comm, gkspc, gwfn)
            else:
                gkspc = GkSpace(gwfn, k_cryst)

            if is_noncolin:
                kswfn = [
                    KSWfn(gkspc, k_weight, 2 * numbnd, is_noncolin),
                ]
                wfn_init(ik, kswfn)
                ksham = [None]
            elif is_spin:
                kswfn = [
                    KSWfn(gkspc, k_weight, numbnd, is_noncolin),
                    KSWfn(gkspc, k_weight, numbnd, is_noncolin),
                ]
                wfn_init(ik, kswfn)
                ksham = [None, None]
            else:
                kswfn = [
                    KSWfn(gkspc, 2 * k_weight, numbnd, is_noncolin),
                ]
                wfn_init(ik, kswfn)
                ksham = [None]
            l_kswfn_kgrp.append(kswfn)
            l_ksham_kgrp.append(ksham)

        if is_gwfn_dist:
            FieldG_rho: FieldGType = get_DistFieldG(grho)
        else:
            FieldG_rho: FieldGType = get_FieldG(grho)
        v_ion, rho_core = FieldG_rho.zeros(()), FieldG_rho.zeros(1)
        l_nloc = []
        for sp in crystal.l_atoms:
            v_ion_sp, rho_core_sp = loc_generate_pot_rhocore(sp, grho)
            v_ion += v_ion_sp
            rho_core += rho_core_sp
            l_nloc.append(NonlocGenerator(sp, gwfn))
        v_ion = v_ion.to_r()

        en = EnergyData()
        if occ_typ == "smear":
            en.fermi, en.smear, en.internal = 0, 0, 0
        elif occ_typ == "fixed":
            en.HO_level, en.LU_level = 0, 0

        if libxc_func is None:
            libxc_func = xc.get_libxc_func(crystal)
        else:
            xc.check_libxc_func(libxc_func)

        # SCF iteration input and output charge densities
        rho_in, rho_out = rho_start, FieldG_rho.empty(1 + is_spin)

        # Defining local potential calculation subroutine
        v_hart: FieldRType
        v_xc: FieldRType
        vloc: FieldRType
        v_ion_g0: Number
        vloc_g0: list[Number]

        def normalize_rho(rho: FieldGType):
            rho *= crystal.numel / (sum(rho.data_g0) * rho.gspc.reallat_dv)

        def compute_vloc():
            nonlocal rho_in, v_hart, v_xc, vloc, vloc_g0, v_ion_g0
            v_hart, en.hartree = hartree.compute(rho_in)

            if hf:
                # Pure HF: no semilocal XC potential or energy
                v_xc = 0.0 * v_hart
                en.xc = 0.0
            else:
                v_xc, en.xc = xc.compute(rho_in, rho_core, *libxc_func)

            vloc = v_ion + v_hart + v_xc

            # Grid interpolation from grho -> gwfn goes here
            vloc /= np.prod(gwfn.grid_shape)

            pwgrp_inter_kgrp.bcast(vloc.data)
            image_comm.bcast(vloc.data)
            v_ion_g0 = np.sum(v_ion) / np.prod(grho.grid_shape)

        # Defining KS Hamiltonian solver routines
        if dftconfig.eigsolve_method == "davidson":
            solver = eigsolve.davidson
            solver_kwargs = {
                "numwork": dftconfig.davidson_numwork,
                "maxiter": dftconfig.davidson_maxiter,
                "vloc_g0": None,
            }
        elif dftconfig.eigsolve_method == "scipy":
            solver = eigsolve.scipy_eigsh
            solver_kwargs = {}
        elif dftconfig.eigsolve_method == "primme":
            solver = eigsolve.primme_eigsh
            solver_kwargs = {
                "numwork": dftconfig.davidson_numwork,
                "maxiter": dftconfig.davidson_maxiter,
                "vloc_g0": None,
            }

        vloc_g0 = None
        exx_drivers = None

        def solve_kswfn(kswfn_k: list[KSWfn], ksham_k: list[KSHam]):
            numiter = 0
            nspin_loop = 2 if is_spin and not is_noncolin else 1

            for ispin in range(nspin_loop):
                kswfn_ = kswfn_k[ispin]

                if ksham_k[ispin] is None:
                    ex = None
                    if hf:
                        # Exchange occupation prefactor
                        occ_spin_factor = (2.0 if (not is_spin and not is_noncolin) else 1.0)
                        f_occ = occ_spin_factor * np.asarray(kswfn_.occ)

                        lat = crystal.reallat.latvec
                        volume = float(abs(np.linalg.det(lat)))

                        Rc_this = None
                        if exxdiv_treatment == "vcut_spherical":
                            if exx_Rc is not None:
                                Rc_this = float(exx_Rc)
                            else:
                                Rc_this = 0.45 * min(
                                    np.linalg.norm(crystal.reallat.latvec[:, 0]),
                                    np.linalg.norm(crystal.reallat.latvec[:, 1]),
                                    np.linalg.norm(crystal.reallat.latvec[:, 2]),
                                )

                        if kpts.numkpts == 1:
                            # Single-k-point exchange
                            ex = Exchange(
                                kswfn_.gkspc,
                                kswfn_.evc_gk,
                                f_occ,
                                volume,
                                method=exxdiv_treatment,
                                Rc=Rc_this,
                            )
                        else:
                            # Multi-k-point exchange
                            nonlocal exx_drivers
                            if exx_drivers is None:
                                nspin_loop_local = 2 if is_spin and not is_noncolin else 1

                                # For non-spin-polarized calculations, k_weight
                                # includes the spin degeneracy factor.
                                kweight_bz_factor = (0.5 if (not is_spin and not is_noncolin) else 1.0)

                                exx_drivers = [
                                    MultiKExchangeDriver(
                                        volume=volume,
                                        nk_total=kpts.numkpts,
                                        exxdiv_treatment=exxdiv_treatment,
                                        exx_Rc=Rc_this,
                                        occ_spin_factor=occ_spin_factor,
                                        kweight_bz_factor=kweight_bz_factor,
                                        use_ace=False,
                                        ace_jitter=1e-12,
                                        kroot_comm=kroot_intra,
                                    )
                                    for _ in range(nspin_loop_local)
                                ]

                                # Initialize the occupied wavefunctions for exchange.
                                for ispin0 in range(nspin_loop_local):
                                    exx_drivers[ispin0].update_from_kswfns(
                                        [kswfn_k[ispin0] for kswfn_k in l_kswfn_kgrp],
                                        build_ace=True,
                                    )

                            ex = exx_drivers[ispin].block_for_kswfn(kswfn_)

                    ksham_k[ispin] = KSHam(
                        kswfn_.gkspc,
                        is_noncolin,
                        vloc if (is_noncolin or (not is_spin)) else vloc[ispin],
                        l_nloc,
                        exchange=ex,
                    )

                ksham_ = ksham_k[ispin]
                ksham_.vloc = vloc if (is_noncolin or (not is_spin)) else vloc[ispin]
                ksham_.vloc_g0 = vloc_g0

                solver_kwargs["vloc_g0"] = vloc_g0
                _, niter = solver.solve(
                    dftcomm, ksham_, kswfn_, diago_thr, **solver_kwargs
                )
                numiter += niter

            return numiter / nspin_loop
        # Generating rho from l_kswfn_kgrp
        def update_rho_out():
            nonlocal rho_out
            rho_wfn = FieldG_rho.zeros(1 + is_spin)
            sl_bnd = scatter_slice(numbnd, n_bgrp, i_bgrp)
            if sl_bnd.start < sl_bnd.stop:
                for kswfn_k in l_kswfn_kgrp:
                    for ispin, kswfn in enumerate(kswfn_k):
                        k_weight = kswfn.k_weight
                        rho_k_spin = kswfn.compute_rho(sl_bnd).to_g()
                        if is_noncolin:
                            rho_wfn[:] += k_weight * rho_k_spin
                        else:
                            rho_wfn[ispin] += k_weight * rho_k_spin
                with dftcomm.pwgrp_inter_image as comm:
                    comm.Allreduce(comm.IN_PLACE, rho_wfn.data)

            # Grid interpolation from gwfn -> grho goes here
            rho_out[:] = rho_wfn[:]
            normalize_rho(rho_out)
            if symm_rho:
                rho_out = symm_mod.symmetrize(rho_out)

        # Defining energy calculation routine
        if is_gwfn_dist:
            # For memory efficiency, compute ewald energy on the rank 0 process, then broadcast
            en.ewald = image_comm.bcast(
                ewald.compute(crystal, grho.gspc_glob) if image_comm.rank == 0 else None
            )
        else:
            en.ewald = image_comm.bcast(ewald.compute(crystal, grho))

        def compute_en():
            nonlocal rho_in, rho_out

            # Eigenvalue sum
            e_eigen = 0.0
            for kswfn_k in l_kswfn_kgrp:
                e_eigen += sum(
                    kswfn_.k_weight * np.sum(kswfn_.evl * kswfn_.occ)
                    for kswfn_ in kswfn_k
                )

            with kroot_intra:
                if kroot_intra.is_null:
                    kroot_intra.skip_with_block()
                e_eigen = kroot_intra.allreduce(e_eigen)

            e_eigen = image_comm.bcast(e_eigen)

            # Energy
            if hf:
                # Exact-exchange energy
                exx = 0.0
                for kswfn_k, ksham_k in zip(l_kswfn_kgrp, l_ksham_kgrp):
                    for ispin in range(2 if is_spin and not is_noncolin else 1):
                        ksh = ksham_k[ispin]
                        if (ksh is not None) and (ksh.exchange is not None):
                            exx += ksh.exchange.energy()

                with kroot_intra:
                    if kroot_intra.is_null:
                        kroot_intra.skip_with_block()
                    exx = kroot_intra.allreduce(exx)

                exx = image_comm.bcast(exx)

                en.exx = float(np.real(exx))
                en.xc = 0.0

                # HF total energy
                en.total = float(np.real(e_eigen - en.hartree - en.exx + en.ewald))
                en.one_el = float(np.real(e_eigen))
                en.hwf = en.total

            else:
                v_hxc = v_hart + v_xc
                e_vhxc = sum((v_hxc * rho_out.to_r())).integrate_unitcell()
                en.one_el = (e_eigen - e_vhxc).real
                en.total = en.one_el + en.ewald + en.hartree + en.xc

                en.hwf = np.real(
                    e_eigen
                    - sum((v_hxc * rho_in.to_r())).integrate_unitcell()
                    + en.ewald
                    + en.hartree
                    + en.xc
                )

            if occ_typ == "smear":
                en.total += en.smear
                en.hwf += en.smear
                en.internal = en.total - en.smear

        # Defining charge mixing routine
        def make_mixmod(rho_ref):
            if dftconfig.mixing_method == "modbroyden":
                return mixing.ModBroyden(dftcomm, rho_ref, mix_beta, mix_dim)
            elif dftconfig.mixing_method == "genbroyden":
                return mixing.GenBroyden(dftcomm, rho_ref, mix_beta, mix_dim)
            else:
                raise ValueError(f"Unknown mixing method: {dftconfig.mixing_method}")

        if symm_rho:
            rho_start = symm_mod.symmetrize(rho_start)

        rho_in = rho_start.copy()
        mixmod = make_mixmod(rho_in)
        image_comm.barrier()

        diago_thr = diago_thr_init
        scf_converged = False
        global_inner_iter = 0

        def occ_factor(kswfn_: KSWfn):
            return (2.0 if (not is_spin and not is_noncolin) else 1.0) * np.asarray(kswfn_.occ)

        def update_exchange_from_wfns(*, build_ace: bool = True):
            """Update the occupied wavefunctions used to construct Vx."""
            nspin_loop = 2 if is_spin and not is_noncolin else 1

            # Update multi-k exchange blocks.
            if hf and (kpts.numkpts != 1):
                nonlocal exx_drivers
                if exx_drivers is None:
                    raise RuntimeError(
                        "exx_drivers is None in multi-k HF update; exchange blocks were not initialized."
                    )
                for ispin0 in range(nspin_loop):
                    exx_drivers[ispin0].update_from_kswfns(
                        [kswfn_k[ispin0] for kswfn_k in l_kswfn_kgrp],
                        build_ace=build_ace,
                    )
                return

            for kswfn_k, ksham_k in zip(l_kswfn_kgrp, l_ksham_kgrp):
                for ispin in range(nspin_loop):
                    ksh = ksham_k[ispin]
                    if (ksh is None) or (ksh.exchange is None):
                        continue

                    kswfn_ = kswfn_k[ispin]
                    ksh.exchange.update_orbitals(
                        kswfn_.evc_gk,
                        occ_factor(kswfn_),
                        build_ace=build_ace,
                    )


        def compute_fock_energy_from_wfns():
            """Return fock = sum_i f_i <psi_i|Vx|psi_i> using the current Vx."""
            fock_val = 0.0
            nspin_loop = 2 if is_spin and not is_noncolin else 1

            for kswfn_k, ksham_k in zip(l_kswfn_kgrp, l_ksham_kgrp):
                for ispin in range(nspin_loop):
                    ksh = ksham_k[ispin]
                    if (ksh is None) or (ksh.exchange is None):
                        continue

                    kswfn_ = kswfn_k[ispin]
                    fock_val += float(
                        np.real(
                            ksh.exchange.fock_energy(
                                kswfn_.evc_gk,
                                occ_factor(kswfn_),
                            )
                        )
                    )

            with kroot_intra:
                if kroot_intra.is_null:
                    kroot_intra.skip_with_block()
                fock_val = kroot_intra.allreduce(fock_val)

            fock_val = image_comm.bcast(fock_val)
            return float(np.real(fock_val))


        def run_inner_scf(conv_thr_target: float):
            nonlocal rho_in, rho_out, diago_thr, scf_converged, global_inner_iter, mixmod

            idxiter = 0
            inner_converged = False
            last_e_error = np.inf

            while idxiter < maxiter:
                print(end="", flush=True)

                if symm_rho:
                    rho_in = symm_mod.symmetrize(rho_in)
                normalize_rho(rho_in)

                compute_vloc()

                # Davidson expects an indexable vloc_g0
                vloc_g0_local = np.sum(vloc, axis=-1) / np.prod(grho.grid_shape)
                if (not is_noncolin) and (not is_spin):
                    vloc_g0_local = (vloc_g0_local,)

                nonlocal vloc_g0
                vloc_g0 = vloc_g0_local

                diago_avgiter = 0.0
                for kswfn_, ksham_ in zip(l_kswfn_kgrp, l_ksham_kgrp):
                    diago_avgiter += solve_kswfn(kswfn_, ksham_)
                diago_avgiter /= len(i_kpts_kgrp)

                if occ_typ == "fixed":
                    en.HO_level, en.LU_level = occup.fixed.compute_occ(
                        dftcomm, l_kswfn_kgrp, crystal.numel
                    )
                elif occ_typ == "smear":
                    en.fermi, en.smear = occup.smear.compute_occ(
                        dftcomm, l_kswfn_kgrp, crystal.numel, is_spin, smear_typ, e_temp
                    )

                update_rho_out()
                compute_en()

                e_error = float(mixmod.compute_error(rho_in, rho_out))
                e_error = image_comm.bcast(e_error)
                last_e_error = e_error

                if e_error < conv_thr_target:
                    inner_converged = True

                    if hf:
                        rho_in = rho_out.copy()
                        if symm_rho:
                            rho_in = symm_mod.symmetrize(rho_in)
                        normalize_rho(rho_in)
                    else:
                        # Preserve the original SCF convergence behaviour.
                        rho_out = rho_in.copy()
                        compute_en()
                elif idxiter == 0 and e_error < diago_thr * crystal.numel:
                    diago_thr = 0.1 * e_error / crystal.numel
                else:
                    diago_thr = min(diago_thr, 0.1 * e_error / crystal.numel)
                    diago_thr = max(diago_thr, 0.5e-13)
                    rho_in = mixmod.mix(rho_in, rho_out)
                    if symm_rho:
                        rho_in = symm_mod.symmetrize(rho_in)

                inner_converged = image_comm.bcast(inner_converged)
                diago_thr = image_comm.bcast(diago_thr)
                image_comm.barrier()

                if iter_printer is not None and image_comm.rank == 0:
                    iter_printer(
                        global_inner_iter,
                        perf_counter() - start_time,
                        inner_converged,
                        e_error,
                        diago_thr,
                        diago_avgiter,
                        en,
                    )

                global_inner_iter += 1

                if inner_converged:
                    break

                idxiter += 1

            return inner_converged, last_e_error

        # HF/EXX outer and inner SCF loops.
        # The exchange operator is fixed during the inner SCF loop and
        # updated between outer EXX iterations.
        if not hf:
            scf_converged, _final_rho_error = run_inner_scf(conv_thr)

        else:
            scf_converged = False

            # Exchange quantities used for outer-loop convergence:
            #   fock0: value from the previous outer iteration
            #   fock1: current wavefunctions with the fixed exchange operator
            #   fock2: current wavefunctions after updating the exchange operator
            #   dexx = fock1 - 0.5 * (fock0 + fock2)
            fock0 = 0.0
            fock1 = 0.0
            fock2 = 0.0
            dexx = np.inf

            conv_thr_this = conv_thr_init if adaptive_thr else conv_thr
            first_exx_step = True

            exx_step = 0
            while exx_step < exx_maxstep:
                inner_converged, final_rho_error = run_inner_scf(conv_thr_this)

                if first_exx_step:
                    # Initialize the exchange operator for the outer loop.
                    update_exchange_from_wfns(build_ace=True)
                    fock2 = compute_fock_energy_from_wfns()
                    en.exx = 0.5 * fock2

                    if image_comm.rank == 0:
                        print(
                            f"[EXX outer {exx_step:02d}] INIT   "
                            f"fock2 = {fock2: .12e}  "
                            f"Ex = {0.5 * fock2: .12e}  "
                            f"conv_thr_inner = {conv_thr_this: .3e}  "
                            f"rho_error = {final_rho_error: .3e}"
                        )

                    first_exx_step = False

                    if reset_mixing_on_exx:
                        mixmod = make_mixmod(rho_in)

                    exx_step += 1
                    # Keep the initial inner threshold for the next cycle.
                    continue

                # Current wavefunctions with the fixed exchange operator.
                fock1 = compute_fock_energy_from_wfns()

                # Update the exchange operator and compute fock2.
                fock0 = fock2
                update_exchange_from_wfns(build_ace=True)
                fock2 = compute_fock_energy_from_wfns()
                en.exx = 0.5 * fock2

                dexx = float(np.real(fock1 - 0.5 * (fock0 + fock2)))

                # Allow small negative values from numerical noise.
                dexx_neg_tol = max(
                    5.0 * exx_conv_thr,
                    1e-10 * max(abs(fock0), abs(fock1), abs(fock2), 1.0),
                )

                if dexx < -dexx_neg_tol:
                    raise RuntimeError(
                        f"dexx is significantly negative: dexx={dexx:.12e}, "
                        f"tolerance={dexx_neg_tol:.12e}. Check the EXX divergence "
                        "treatment, k-point weights, and EXX cutoff."
                    )

                if dexx < 0.0:
                    if image_comm.rank == 0:
                        print(
                            f"[EXX outer {exx_step:02d}] small negative dexx clipped to zero: "
                            f"{dexx: .12e}  (tol = {dexx_neg_tol: .12e})"
                        )
                    dexx = 0.0

                if image_comm.rank == 0:
                    print(
                        f"[EXX outer {exx_step:02d}] STEP   "
                        f"fock0 = {fock0: .12e}  "
                        f"fock1 = {fock1: .12e}  "
                        f"fock2 = {fock2: .12e}  "
                        f"dexx = {dexx: .3e}  "
                        f"conv_thr_inner = {conv_thr_this: .3e}  "
                        f"rho_error = {final_rho_error: .3e}"
                    )

                if inner_converged and np.isfinite(dexx) and (dexx < exx_conv_thr):
                    scf_converged = True
                    break

                if adaptive_thr:
                    if np.isfinite(dexx):
                        conv_thr_this = max(conv_thr, conv_thr_multi * dexx)
                    else:
                        conv_thr_this = conv_thr_init
                else:
                    conv_thr_this = conv_thr

                if reset_mixing_on_exx:
                    mixmod = make_mixmod(rho_in)

                exx_step += 1

        if scf_converged and image_comm.rank == 0:
            print("SCF Converged.")
            if is_spin:
                tot_mag = rho_out[0].to_r() - rho_out[1].to_r()
                abs_mag = tot_mag.copy()
                abs_mag._data[:] = np.abs(tot_mag.data)
                print(
                    f"Total magnetization:    {tot_mag.integrate_unitcell().real:>5.2f} Bohr magneton / cell (Ry units)"
                )
                print(
                    f"Absolute magnetization: {abs_mag.integrate_unitcell().real:>5.2f} Bohr magneton / cell (Ry units)"
                )

        # TODO: Make it a separate function.
        if ret_vxc:

            def calculate_vxc_data():
                vxc_arr = np.zeros((len(l_kswfn_kgrp), numbnd))
                v_xc, en.xc = xc.compute(rho_in, rho_core, *libxc_func)
                for ik, kswfn in enumerate(l_kswfn_kgrp):
                    wfn = kswfn[0]
                    psi_r_allbands = wfn.evc_gk.to_r()
                    for iband in range(wfn.numbnd):
                        psi_r = psi_r_allbands[iband]
                        hpsi_r = sum((v_xc * psi_r.get_density())).integrate_unitcell()
                        vxc_arr[ik, iband] = hpsi_r.real
                return np.array(vxc_arr)

            vxc_arr = calculate_vxc_data()
            return scf_converged, rho_in, l_kswfn_kgrp, en, vxc_arr
        else:
            return scf_converged, rho_in, l_kswfn_kgrp, en
