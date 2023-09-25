from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal
    from numbers import Number
__all__ = ['scf', 'EnergyData', 'IterPrinter']

from dataclasses import dataclass
from time import perf_counter
from sys import version_info
import numpy as np

from qtm.crystal import Crystal
from qtm.kpts import KList
from qtm.gspace import GSpace, GkSpace
from qtm.containers import FieldGType, FieldRType, get_FieldG, get_FieldR

from qtm.pot import hartree, xc, ewald
from qtm.pseudo import (
    loc_generate_rhoatomic, loc_generate_pot_rhocore,
    NonlocGenerator
)
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


if version_info[1] >= 8:
    from typing import Protocol

    class IterPrinter(Protocol):
        def __call__(self, idxiter: int, runtime: float, scf_converged: bool,
                     e_error: float, diago_thr: float, diago_avgiter: float,
                     en: EnergyData) -> None:
            ...

    class WfnInit(Protocol):
        def __call__(self, ik: int, kswfn: list[KSWfn]) -> None:
            ...
else:
    IterPrinter = 'IterPrinter'
    WfnInit = 'WfnInit'


def scf(dftcomm: DFTCommMod, crystal: Crystal, kpts: KList,
        grho: GSpace, gwfn: GSpace, numbnd: int,
        is_spin: bool, is_noncolin: bool,
        symm_rho: bool = True, 
        rho_start: FieldGType | tuple[float, ...] | None = None,
        wfn_init: WfnInit | None = None,
        libxc_func: tuple[str, str] | None = None,
        occ_typ: Literal['fixed', 'smear'] = 'smear',
        smear_typ: Literal['gauss', 'fd', 'mv'] = 'gauss',
        e_temp: float = 1E-3,
        conv_thr: float = 1E-6*RYDBERG, maxiter: int = 100,
        diago_thr_init: float = 1E-2*RYDBERG,
        iter_printer: IterPrinter | None = None,
        mix_beta: float = 0.7, mix_dim: int = 8,
        dftconfig: DFTConfig | None = None,
        ret_vxc:bool=False
        ):
    if not isinstance(dftcomm, DFTCommMod):
        raise TypeError(
            type_mismatch_msg('dftcomm', dftcomm, DFTCommMod)
        )

    # Checking input arguments
    with dftcomm.image_comm as comm:
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
            raise NotImplementedError(
                "noncollinear calculations yet to be implemented"
            )
        if is_noncolin:
            is_spin = True

        symm_rho = comm.bcast(symm_rho)
        assert isinstance(symm_rho, bool)

        if isinstance(rho_start, FieldGType):
            assert rho_start.gspc is grho
            if rho_start.gspc is not grho:
                raise ValueError(
                    obj_mismatch_msg('rho_start.gspc', rho_start.gspc, 'grho', grho)
                )
            if rho_start.shape != (1 + is_spin, ):
                raise ValueError(value_mismatch_msg(
                    'rho_start.shape', rho_start.shape,
                    f'(1 + is_spin, ) = {(1 + is_spin, )}'
                ))
        elif is_spin:
            if rho_start is None:
                raise ValueError(type_mismatch_msg(
                    'rho_start', rho_start,
                    f"a {get_FieldG} instance or a sequeuce of numbers between -1 and +1 "
                    f"representing starting spin polarisation on each atomic type of"
                    f"crystal (for is_spin = {is_spin})"
                ))
            starting_mag = rho_start
            rho_start = get_FieldG(grho).zeros(1 + is_spin)
            for sp, mag in zip(crystal.l_atoms, starting_mag):
                rho_atomic_sp = loc_generate_rhoatomic(sp, grho)
                mag = max(min(mag, 1), -1)
                rho_start[0] = ((1 + mag) / 2) * rho_atomic_sp
                rho_start[1] = ((1 - mag) / 2) * rho_atomic_sp
        else:
            rho_start = sum(loc_generate_rhoatomic(sp, grho)
                            for sp in crystal.l_atoms).reshape(1)

        libxc_func = comm.bcast(libxc_func)

        occ_typ = comm.bcast(occ_typ)
        assert occ_typ in ['fixed', 'smear']

        smear_typ = comm.bcast(smear_typ)
        assert smear_typ in ['gauss', 'fd', 'mv']

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

    start_time = perf_counter()
    image_comm = dftcomm.image_comm
    pwgrp_inter_kgrp = dftcomm.pwgrp_inter_kgrp
    kroot_intra = dftcomm.kroot_intra

    with image_comm:
        n_kgrp, i_kgrp = dftcomm.n_kgrp, dftcomm.i_kgrp
        n_bgrp, i_bgrp = dftcomm.n_bgrp, dftcomm.i_bgrp
        i_kpts_kgrp = range(kpts.numkpts)[
            scatter_slice(kpts.numkpts, n_kgrp, i_kgrp)
        ]

        symm_mod = None
        if symm_rho:
            symm_mod = SymmFieldMod(crystal, grho)

        if wfn_init is None:
            def wfn_init(_, kswfn_k):
                kswfn_k[0].init_random()
                if is_spin and not is_noncolin:
                    kswfn_k[1].init_random()

        l_kswfn_kgrp = []

        for ik in i_kpts_kgrp:
            k_cryst, k_weight = kpts[ik]
            gkspc = GkSpace(gwfn, k_cryst)
            if is_noncolin:
                kswfn = [
                    KSWfn(gkspc, k_weight, 2 * numbnd, is_noncolin),
                ]
                wfn_init(ik, kswfn)
            elif is_spin:
                kswfn = [
                    KSWfn(gkspc, k_weight, numbnd, is_noncolin),
                    KSWfn(gkspc, k_weight, numbnd, is_noncolin)
                ]
                wfn_init(ik, kswfn)
            else:
                kswfn = [
                    KSWfn(gkspc, 2 * k_weight, numbnd, is_noncolin),
                ]
                wfn_init(ik, kswfn)
            l_kswfn_kgrp.append(kswfn)

        FieldG_rho: FieldGType = get_FieldG(grho)
        v_ion, rho_core = FieldG_rho.zeros(()), FieldG_rho.zeros(1)
        l_nloc = []
        for sp in crystal.l_atoms:
            v_ion_sp, rho_core_sp = loc_generate_pot_rhocore(sp, grho)
            v_ion += v_ion_sp
            rho_core += rho_core_sp
            l_nloc.append(NonlocGenerator(sp, gwfn.ecut / 4))
        v_ion = v_ion.to_r()

        en = EnergyData()
        if occ_typ == 'smear':
            en.fermi, en.smear, en.internal = 0, 0, 0
        elif occ_typ == 'fixed':
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
            v_xc, en.xc = xc.compute(rho_in, rho_core, *libxc_func)
            vloc = v_ion + v_hart + v_xc

            # Grid interpolation from grho -> gwfn goes here
            vloc /= np.prod(gwfn.grid_shape)
            vloc_g0 = np.sum(vloc, axis=-1)
            pwgrp_inter_kgrp.bcast(vloc.data)
            image_comm.bcast(vloc.data)
            v_ion_g0 = np.sum(v_ion) / np.prod(grho.grid_shape)


        # Defining KS Hamiltonian solver routines
        if dftconfig.eigsolve_method == 'davidson':
            solver = eigsolve.davidson
            solver_kwargs = {'numwork': dftconfig.davidson_numwork,
                             'maxiter': dftconfig.davidson_maxiter,
                             'vloc_g0': None}
        elif dftconfig.eigsolve_method == 'scipy':
            solver = eigsolve.scipy_eigsh
            solver_kwargs = {}

        def solve_kswfn(kswfn_k: list[KSWfn]):
            numiter = 0
            for ispin in range(2 if is_spin and not is_noncolin else 1):
                kswfn_ = kswfn_k[ispin]
                ksham = KSHam(kswfn_.gkspc, is_noncolin,
                              vloc if is_noncolin else vloc[ispin], l_nloc)
                solver_kwargs['vloc_g0'] = vloc_g0
                _, niter = solver.solve(dftcomm, ksham, kswfn_, diago_thr,
                                        **solver_kwargs)
                numiter += niter
            numiter /= 2 if is_spin and not is_noncolin else 1
            return numiter

        # Generating rho from l_kswfn_kgrp
        def update_rho_out():
            nonlocal rho_out
            rho_wfn = FieldG_rho.zeros(1 + is_spin)
            sl_bnd = scatter_slice(numbnd, n_bgrp, i_bgrp)
            if sl_bnd.start < sl_bnd.stop:
                for kswfn_k in l_kswfn_kgrp:
                    for ispin, kswfn in enumerate(kswfn_k):
                        k_weight = kswfn.k_weight
                        rho_k_spin = kswfn.compute_rho(sl_bnd)
                        if is_noncolin:
                            rho_wfn[:] += k_weight * rho_k_spin.to_g()
                        else:
                            rho_wfn[ispin] += k_weight * rho_k_spin.to_g()
                with dftcomm.pwgrp_inter_image as comm:
                    comm.Allreduce(comm.IN_PLACE, rho_wfn.data)

            # Grid interpolation from gwfn -> grho goes here
            rho_out[:] = rho_wfn[:]
            normalize_rho(rho_out)
            if symm_rho:
                rho_out = symm_mod.symmetrize(rho_out)

        # Defining energy calculation routine
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
            en.hwf = e_eigen - sum((v_hxc * rho_in.to_r())).integrate_unitcell()
            if occ_typ == 'smear':
                en.total += en.smear
                en.hwf += en.smear
                en.internal = en.total - en.smear

        # Defining charge mixing routine
        # if dftconfig.mixing_method == 'modbroyden':
        if symm_rho:
            rho_start = symm_mod.symmetrize(rho_start)
        mixmod = mixing.ModBroyden(dftcomm, rho_start, mix_beta, mix_dim)

        diago_thr = diago_thr_init
        scf_converged = False

        idxiter = 0
        while idxiter < maxiter:
            print(end="",flush=True)
            normalize_rho(rho_in)
            compute_vloc()
            vloc_g0 = np.sum(vloc, axis=-1) / np.prod(grho.grid_shape)

            diago_avgiter = 0
            for kswfn_ in l_kswfn_kgrp:
                diago_avgiter += solve_kswfn(kswfn_)
            diago_avgiter /= len(i_kpts_kgrp)
            if occ_typ == 'fixed':
                en.HO_level, en.LU_level = occup.fixed.compute_occ(
                    dftcomm, l_kswfn_kgrp, crystal.numel
                )
            elif occ_typ == 'smear':
                en.fermi, en.smear = occup.smear.compute_occ(
                    dftcomm, l_kswfn_kgrp, crystal.numel, is_spin, smear_typ, e_temp
                )

            update_rho_out()
            compute_en()
            e_error = mixmod.compute_error(rho_in, rho_out)
            e_error = image_comm.bcast(e_error)

            if e_error < conv_thr:
                scf_converged = True
            elif idxiter == 0 and e_error < diago_thr * crystal.numel:
                if iter_printer is not None:
                    iter_printer(
                        idxiter, perf_counter() - start_time, scf_converged,
                        e_error, diago_thr, diago_avgiter, en
                    )
                diago_thr = 0.1 * e_error / crystal.numel
                continue
            else:
                diago_thr = min(diago_thr, 0.1 * e_error / crystal.numel)
                diago_thr = max(diago_thr, 1E-13)
                rho_in = mixmod.mix(rho_in, rho_out)
                if symm_rho:
                    rho_in = symm_mod.symmetrize(rho_in)

            scf_converged = image_comm.bcast(scf_converged)
            diago_thr = image_comm.bcast(diago_thr)
            image_comm.barrier()
            if iter_printer is not None and image_comm.rank == 0:
                iter_printer(
                    idxiter, perf_counter() - start_time, scf_converged,
                    e_error, diago_thr, diago_avgiter, en
                )
            if scf_converged:
                break
            else:
                idxiter += 1

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
                        vxc_arr[ik,iband] = hpsi_r
                return np.array(vxc_arr)
            
            vxc_arr = calculate_vxc_data()
            return scf_converged, rho_in, l_kswfn_kgrp, en, vxc_arr
        else:
            return scf_converged, rho_in, l_kswfn_kgrp, en
