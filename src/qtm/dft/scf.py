from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import gc

from qtm.logger import qtmlogger
from qtm.config import MPI4PY_INSTALLED

if MPI4PY_INSTALLED:
    from qtm.mpi.containers import get_DistFieldG
from qtm.mpi.gspace import DistGSpace, DistGkSpace

if TYPE_CHECKING:
    from typing import Literal
    from numbers import Number
__all__ = ["scf", "EnergyData", "Iterprinter"]

from dataclasses import dataclass
from time import perf_counter
from sys import version_info
import numpy as np

from qtm.crystal import Crystal
from qtm.kpts import KList
from qtm.gspace import GSpace, GkSpace
from qtm.containers import FieldGType, FieldRType, get_FieldG

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

    fermi: float | None = None
    smear: float | None = None
    internal: float | None = None

    HO_level: float | None = None
    LU_level: float | None = None


from qtm.io_utils.dft_printers import print_scf_parameters


#region Type hinting for Python 3.8 and above
if version_info[1] >= 8:
    from typing import Protocol

    class Iterprinter(Protocol):
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
    Iterprinter = "Iterprinter"
    WfnInit = "WfnInit"
#endregion

#region SCF function is starting here
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
    iter_printer: Iterprinter | None = None,
    mix_beta: float = 0.3,
    mix_dim: int = 8,
    dftconfig: DFTConfig | None = None,
    ret_vxc: bool = False,
    force_stress: bool = False,
) -> (
    tuple[bool, FieldGType, list[list[KSWfn]], EnergyData, np.ndarray]
    | tuple[bool, FieldGType, list[list[KSWfn]], EnergyData]
):
    #region Checking DFTCOMMMod
    if not isinstance(dftcomm, DFTCommMod):
        raise TypeError(type_mismatch_msg("dftcomm", dftcomm, DFTCommMod))

    #endregion

    with dftcomm.image_comm as comm:
        #region Checking input arguments
        check_system(comm, crystal, grho, gwfn, kpts)
        if grho is not gwfn:
            raise NotImplementedError()

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
        #endregion

        #region checking the input rho_start
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
        #endregion

        #region checking the starting rho for spin polarization
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
        #endregion

        #region checking the libxc_func
        libxc_func = comm.bcast(libxc_func)
        #endregion

        #region checking the other input parameteres: occ_typ, smear_typ, e_temp, conv_thr, maxiter, diago_thr_init, mix_beta, mix_dim, dftconfig   
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

        #endregion

    #region making the groups for k points, g space and bands
    start_time = perf_counter()
    image_comm = dftcomm.image_comm
    pwgrp_inter_kgrp = dftcomm.pwgrp_inter_kgrp
    kroot_intra = dftcomm.kroot_intra
    #endregion

    #print("Starting SCF calculation", flush=True)

    #region Checking the number of k-points and k-groups
    assert (
        kpts.numkpts >= dftcomm.n_kgrp
    ), "Number of k-points must be greater than or equal to the number of k-groups"
    #endregion

    #region #printing the SCF parameters
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
    #endregion

    #print("#printed scf parameters", flush=True)   
    #region the main definitions across all the processors will start here
    with image_comm:
        #region defining the parallelization groups
        n_kgrp, i_kgrp = dftcomm.n_kgrp, dftcomm.i_kgrp
        n_bgrp, i_bgrp = dftcomm.n_bgrp, dftcomm.i_bgrp
        i_kpts_kgrp = range(kpts.numkpts)[scatter_slice(kpts.numkpts, n_kgrp, i_kgrp)]    ##This represents the k points in each group
        #endregion

        #region defining the symmetry group
        symm_mod = None
        # if symm_rho:
        symm_mod = SymmFieldMod(crystal, grho)
        #endregion

        #region defining the wavefunction initialization
        if wfn_init is None:

            def wfn_init(ik, kswfn_k):
                np.random.seed(ik)  # For reproducible speed measurements
                kswfn_k[0].init_random()
                if is_spin and not is_noncolin:
                    kswfn_k[1].init_random()
        #endregion

        l_kswfn_kgrp = []
        l_ksham_kgrp = []

        #print("initialized wavefunction container", flush=True)

        #region creatng the g space, gk space, kswfn and ksham containers

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

        #endregion

        #print("initialized gk and g space and ksham wavefunction", flush=True)

        #region defining g space for g space parallelzation
        if is_gwfn_dist:
            FieldG_rho: FieldGType = get_DistFieldG(grho)
        else:
            FieldG_rho: FieldGType = get_FieldG(grho)
        #endregion
        

        #region defining the local potential and core charge
        v_ion, rho_core = FieldG_rho.zeros(()), FieldG_rho.zeros(1)

        v_ion_list=[]
        l_nloc = []
        for sp in crystal.l_atoms:
            v_ion_sp, rho_core_sp, v_ion_nomult, rho_core_nomult = loc_generate_pot_rhocore(sp, grho)
            v_ion += v_ion_sp
            rho_core += rho_core_sp
            l_nloc.append(NonlocGenerator(sp, gwfn))
            v_ion_list.append(v_ion_nomult)
        v_ion = v_ion.to_r()

        #endregion  

        #print("initialized v_ion and rho_core", flush=True)

        #region defining the energy data
        en = EnergyData()
        if occ_typ == "smear":
            en.fermi, en.smear, en.internal = 0, 0, 0
        elif occ_typ == "fixed":
            en.HO_level, en.LU_level = 0, 0
        #endregion

        #region defining the libxc function
        if libxc_func is None:
            libxc_func = xc.get_libxc_func(crystal)
        else:
            xc.check_libxc_func(libxc_func)
        #endregion
        

        #print("initialized energy data", flush=True)

        #region SCF iteration input and output charge densities
        rho_in, rho_out = rho_start, FieldG_rho.empty(1 + is_spin)
        #endregion
        #print("initialized rho_in and rho_out", flush=True)

        #region Defining local potential calculation subroutine
        v_hart: FieldRType
        v_xc: FieldRType
        vloc: FieldRType
        v_ion_g0: Number
        vloc_g0: list[Number]
        xc_compute:list[FieldRType, Number, np.ndarray]
        #endregion

        #region normalizing the charge density
        def normalize_rho(rho: FieldGType):
            rho *= crystal.numel / (sum(rho.data_g0) * rho.gspc.reallat_dv)
        #endregion

        #region computing the local potential
        def compute_vloc():
            nonlocal rho_in, v_hart, v_xc, vloc, vloc_g0, v_ion_g0, xc_compute
            v_hart, en.hartree = hartree.compute(rho_in)
            v_xc, en.xc, GGA = xc.compute(rho_in, rho_core, *libxc_func)
            xc_compute=(v_xc, en.xc, GGA)
            vloc = v_ion + v_hart + v_xc

            # Grid interpolation from grho -> gwfn goes here
            vloc /= np.prod(gwfn.grid_shape)
            vloc_g0 = np.sum(vloc, axis=-1)
            pwgrp_inter_kgrp.bcast(vloc.data)
            image_comm.bcast(vloc.data)
            v_ion_g0 = np.sum(v_ion) / np.prod(grho.grid_shape)
        #endregion

        #region computing the Hartree and exchange-correlation potential
        def v_hxc(rho):
            nonlocal rho_in, v_hxc
            v_hart, en.hartree = hartree.compute(rho)
            v_xc, en.xc, GGA = xc.compute(rho, rho_core, *libxc_func)

            v_hxc_r = v_hart + v_xc
            v_hxc_g=v_hxc_r.to_g()
            v_hxc_g /= np.prod(gwfn.grid_shape)

            return v_hxc_g
        #endregion
        
        #print("initialized vloc", flush=True)

        #region Defining KS Hamiltonian solver routines
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
        #endregion
        
        #print("initialized solver", flush=True)

        #region Defining the subroutine to solve the Kohn-Sham wavefunctions
        def solve_kswfn(kswfn_k: list[KSWfn], ksham_k: list[KSHam],
                        force_stress=force_stress) -> float:
            numiter = 0
            for ispin in range(2 if is_spin and not is_noncolin else 1):
                kswfn_ = kswfn_k[ispin]
                if ksham_k[ispin] is None:
                    ksham_k[ispin] = KSHam(
                        kswfn_.gkspc,
                        is_noncolin,
                        vloc if is_noncolin else vloc[ispin],
                        l_nloc,
                    )
                ksham_ = ksham_k[ispin]
                ksham_.vloc = vloc if is_noncolin else vloc[ispin]
                ksham_.vloc_g0 = vloc_g0
                # ksham = KSHam(kswfn_.gkspc, is_noncolin,
                #               vloc if is_noncolin else vloc[ispin], l_nloc)
                solver_kwargs["vloc_g0"] = vloc_g0
                _, niter = solver.solve(
                    dftcomm, ksham_, kswfn_, diago_thr, **solver_kwargs
                )
                if force_stress and ispin==0:
                    vkb_dij=ksham_.l_vkb_dij
                else: vkb_dij=None 
                numiter += niter
            numiter /= 2 if is_spin and not is_noncolin else 1
            return numiter, vkb_dij
        #endregion
        
        #print("initialized solve_kswfn", flush=True)

        #region Generating rho from l_kswfn_kgrp
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
        #endregion


        #print("initialized update_rho_out", flush=True)

        #region Defining energy calculation routine
        if is_gwfn_dist:
            # For memory efficiency, compute ewald energy on the rank 0 process, then broadcast
            en.ewald = image_comm.bcast(
                ewald.compute(crystal, grho.gspc_glob) if image_comm.rank == 0 else None
            )
        else:
            en.ewald = image_comm.bcast(ewald.compute(crystal, grho))

        def compute_en():
            nonlocal rho_in, rho_out
            e_eigen = 0
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
            )  # Harris-Weinert-Foulkes energy estimate
            if occ_typ == "smear":
                en.total += en.smear
                en.hwf += en.smear
                en.internal = en.total - en.smear
        #endregion

        #print("initialized compute_en", flush=True)

        #region Defining charge mixing routine
        if dftconfig.mixing_method == "modbroyden":
            mixmod = mixing.ModBroyden(dftcomm, rho_start, mix_beta, mix_dim)
        elif dftconfig.mixing_method == "genbroyden":
            mixmod = mixing.GenBroyden(dftcomm, rho_start, mix_beta, mix_dim)
        if symm_rho:
            rho_start = symm_mod.symmetrize(rho_start)
        comm.barrier()
        #endregion

        diago_thr = diago_thr_init
        scf_converged = False

        idxiter = 0
        nloc_dij_vkb=[]

        #print("initialized mixing routine", flush=True)

        #region starting the SCF loop
        while idxiter < maxiter:
            #print("scf iteration is staring, at iteration", idxiter, flush=True)
            #print(end="", flush=True)
            if symm_rho:
                rho_in = symm_mod.symmetrize(rho_in)
            normalize_rho(rho_in)
            #print("normalized rho_in, at iteration", idxiter, flush=True)
            compute_vloc()
            #print("computed vloc, at iteration", idxiter, flush=True)
            v_in_hxc=v_hxc(rho_in)
            vloc_g0 = np.sum(vloc, axis=-1) / np.prod(grho.grid_shape)
            #print("computed vloc_g0, at iteration", idxiter, flush=True)

            diago_avgiter = 0
            for kswfn_, ksham_ in zip(l_kswfn_kgrp, l_ksham_kgrp):
                numiter, dij_vkb = solve_kswfn(kswfn_, ksham_)
                if idxiter==0: nloc_dij_vkb.append(dij_vkb)
                diago_avgiter += numiter
            diago_avgiter /= len(i_kpts_kgrp)
            #print("solved kswfn, at iteration", idxiter, flush=True)

            if occ_typ == "fixed":
                en.HO_level, en.LU_level = occup.fixed.compute_occ(
                    dftcomm, l_kswfn_kgrp, crystal.numel
                )
                #print("computed fixed occupation, at iteration", idxiter, flush=True)
            #print("starting smearing occupation, if needed", flush=True)  
            if occ_typ == "smear":
                #print("now we are going to smear the occupation", flush=True)
                en.fermi, en.smear = occup.smear.compute_occ(
                    dftcomm, l_kswfn_kgrp, crystal.numel, is_spin, smear_typ, e_temp
                )
                #print("computed smear occupation, at iteration", idxiter, flush=True)

            update_rho_out()
            #print("updated rho_out, at iteration", idxiter, flush=True)
            compute_en()
            #print("computed energy, at iteration", idxiter, flush=True)
            v_out_hxc = v_hxc(rho_out)

            del_v_hxc=v_out_hxc - v_in_hxc
            e_error = float(mixmod.compute_error(rho_in, rho_out))
            #print("computed error, at iteration", idxiter, flush=True)
            e_error = image_comm.bcast(e_error)
            #print("broadcasted error, at iteration", idxiter, flush=True)

            if (
                e_error < conv_thr
            ):  # and diago_thr < max(1e-13, 0.1*conv_thr/crystal.numel):
                #print("e_error is less than conv_thr, at iteration", idxiter, "finishing the iteration", flush=True)
                scf_converged = True
                rho_out = rho_in.copy()
                #print("final rho_out", flush=True)
                compute_en()
                #print("final energy", flush=True)
            elif idxiter == 0 and e_error < diago_thr * crystal.numel:
                #print("e_error is less than diago_thr, at iteration", idxiter, "finishing the iteration", flush=True)
                diago_thr = 0.1 * e_error / crystal.numel
                # continue
            else:
                #print("e_error is greater than diago_thr, at iteration", idxiter, "starting the next iteration", flush=True)
                diago_thr = min(diago_thr, 0.1 * e_error / crystal.numel)
                diago_thr = max(diago_thr, 0.5e-13)
                rho_in = mixmod.mix(rho_in, rho_out)
                #print("mixed rho_in and rho_out, at iteration", idxiter, flush=True)
                if symm_rho:
                    rho_in = symm_mod.symmetrize(rho_in)
                #print("symmetrized rho_in, at iteration", idxiter, flush=True)

            scf_converged = image_comm.bcast(scf_converged)
            diago_thr = image_comm.bcast(diago_thr)
            image_comm.barrier()
            if iter_printer is not None and image_comm.rank == 0:
                iter_printer(
                    idxiter,
                    perf_counter() - start_time,
                    scf_converged,
                    e_error,
                    diago_thr,
                    diago_avgiter,
                    en,
                )
            if scf_converged:
                if image_comm.rank == 0:
                    #print("SCF Converged.")
                    if is_spin:
                        # Total magnetization = int rho_up(r)-rho_down(r) dr
                        tot_mag = rho_out[0].to_r() - rho_out[1].to_r()
                        # Abs. magnetization = int |rho_up(r)-rho_down(r)| dr
                        abs_mag = tot_mag.copy()
                        abs_mag._data[:] = np.abs(tot_mag.data)
                        print(
                            f"Total magnetization:    {tot_mag.integrate_unitcell().real:>5.2f} Bohr magneton / cell (Ry units)"
                        )
                        print(
                            f"Absolute magnetization: {abs_mag.integrate_unitcell().real:>5.2f} Bohr magneton / cell (Ry units)"
                        )
                break
            else:
                idxiter += 1

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
            for var in list(locals().keys()):
                if var not in ["scf_converged", "rho_in", "l_kswfn_kgrp", "en", "vxc_arr"]:
                    del locals()[var]
            gc.collect()
            return scf_converged, rho_in, l_kswfn_kgrp, en, vxc_arr
        if force_stress:
            for var in list(locals().keys()):
                if var not in ["scf_converged", "rho_in", "l_kswfn_kgrp", "en", "nloc_dij_vkb"]:
                    del locals()[var]
            gc.collect()
            return scf_converged, rho_in, l_kswfn_kgrp, en, v_ion_list, nloc_dij_vkb, xc_compute, del_v_hxc
        else:
            for var in list(locals().keys()):
                if var not in ["scf_converged", "rho_in", "l_kswfn_kgrp", "en"]:
                    del locals()[var]
            gc.collect()
            return scf_converged, rho_in, l_kswfn_kgrp, en
    #endregion
#endregion