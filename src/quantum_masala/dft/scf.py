__all__ = ["scf"]
from typing import Optional, Callable, Any
import numpy as np

from quantum_masala.core import (
    Crystal,
    KPoints, kpts_distribute,
    GSpace, GField, RField,
    Wavefun,
    rho_check, rho_normalize,
)
from quantum_masala.pseudo import (
    loc_generate, NonlocGenerator
)
from quantum_masala.dft.kswfn import (
    KSWavefun,
    wfn_generate, wfn_gen_rho
)
from quantum_masala.dft.pot import (
    hartree_compute, xc_compute,
    ewald_compute,
)

from quantum_masala.dft.eigsolve import solve_wfn

from quantum_masala.dft.occ import fixed, smear

from quantum_masala.dft.mix import *

from quantum_masala import config, pw_logger

@pw_logger.time('dft:scf')
def scf(crystal: Crystal, kpts: KPoints,
        rho_start: GField, symm_rho: bool,
        numbnd: int, is_spin: bool, is_noncolin: bool,
        wfn_init: Callable[[KSWavefun, int], None],
        xc_params: dict[str, Any],
        occ: str = 'smear', smear_typ: str = 'gauss', e_temp: float = 1E-3,
        conv_thr: float = 1E-6, max_iter: int = 100,
        diago_thr_init: float = 1E-2,
        iter_printer: Optional[
            Callable[[int, bool, float, Optional[float], dict[str, float]], None]] = None,
        mix_beta: float = 0.7, mix_dim: int = 8,
        ):
    pwcomm = config.pwcomm
    numel = crystal.numel

    rho_check(rho_start, is_spin)
    rho_start = rho_normalize(rho_start, numel)
    rho = rho_start.copy()

    if config.mixing_method == 'genbroyden':
        mixmod = GenBroyden(rho, mix_beta, mix_dim)
    elif config.mixing_method == 'modbroyden':
        mixmod = ModBroyden(rho, mix_beta, mix_dim)
    elif config.mixing_method == 'anderson':
        mixmod = Anderson(rho, mix_beta, mix_dim)
    else:
        raise ValueError('abc')

    grho = rho.gspc
    gwfn = grho

    kpts_kgrp, idxkpts_kgrp = kpts_distribute(kpts, True, True)
    numkpts_kgrp = len(idxkpts_kgrp)

    l_wfn_kgrp = wfn_generate(gwfn, kpts, numbnd, is_spin, is_noncolin,
                              idxkpts_kgrp)
    for ikpt in range(numkpts_kgrp):
        wfn_init(l_wfn_kgrp[ikpt], idxkpts_kgrp[ikpt])

    v_ion, rho_core = GField.zeros(grho, 1), GField.zeros(grho, 1)

    l_nloc = []
    for sp in crystal.l_atoms:
        v_ion_typ, rho_core_typ = loc_generate(sp, grho)
        v_ion += v_ion_typ
        rho_core += rho_core_typ
        l_nloc.append(NonlocGenerator(sp, gwfn))
    v_ion.symmetrize()
    v_ion = v_ion.to_rfield()

    rho_out: GField
    v_hart: RField
    v_xc: RField
    v_loc: RField

    en = {'total': 0, 'hwf': 0, 'one_el': 0,
          'hart': 0, 'xc': 0, 'ewald': 0}
    if occ == 'smear':
        en['fermi'], en['smear'] = 0, 0
    elif occ == 'fixed':
        en['occ_max'], en['unocc_min'] = 0, 0

    def compute_pot_local(rho, rho_core):
        nonlocal v_hart, v_xc, v_loc
        v_hart, en['hart'] = hartree_compute(rho)
        v_xc, en['xc'] = xc_compute(rho, rho_core, **xc_params)
        v_loc = v_ion + v_hart + v_xc
        v_loc.Bcast()
        return v_loc

    if config.use_gpu:
        from .gpu.ksham_gpu import KSHamGPU as KSHam
    else:
        from .ksham import KSHam

    def gen_ham(wfn: KSWavefun):
        return KSHam(wfn.gkspc, wfn.is_spin, wfn.is_noncolin, v_loc, l_nloc)

    en['ewald'] = ewald_compute(crystal, grho)

    def compute_energies():
        rho_r = rho.to_rfield()
        rho_out_r = rho_out.to_rfield()
        e_eigen = sum(wfn.k_weight * np.sum(wfn.evl * wfn.occ)
                      for wfn in l_wfn_kgrp)
        if pwcomm.kgrp_rank == 0:
            e_eigen = pwcomm.kgrp_intercomm.allreduce_sum(e_eigen)
        vhxc = v_hart + v_xc
        e_eigen = pwcomm.world_comm.bcast(e_eigen) * (2 if not is_spin else 1)
        en['one_el'] = e_eigen - rho_out_r.integrate(vhxc, axis=0).real
        en['total'] = en['one_el'] + en['ewald'] + en['hart'] + en['xc']
        en['hwf'] = e_eigen - rho_r.integrate(vhxc, axis=0).real \
            + en['ewald'] + en['hart'] + en['xc']
        if occ == 'smear':
            en['int'] = en['total']
            en['total'] += en['smear']
            en['hwf'] += en['smear']


    diago_thr = diago_thr_init
    scf_converged = False
    if symm_rho:
        rho.symmetrize()
    rho = rho_normalize(rho, numel)
    rho.Bcast()

    idxiter = 0
    while idxiter < max_iter:
        v_loc = compute_pot_local(rho, rho_core)

        prec_params = {}
        if config.eigsolve_method == 'davidson':
            prec_params['vbare_g0'] = np.sum(v_ion.r) / np.prod(grho.grid_shape)

        diago_avgiter = 0
        for _wfn in l_wfn_kgrp:
            diago_avgiter += solve_wfn(_wfn, gen_ham, diago_thr, **prec_params)
        diago_avgiter /= numkpts_kgrp * (1 + is_spin)
        if occ == 'fixed':
            en['occ_max'], en['unocc_min'] = fixed.compute_occ(
                l_wfn_kgrp, numel
            )
        elif occ == 'smear':
            en['fermi'], en['smear'] = smear.compute_occ(
                l_wfn_kgrp, numel, smear_typ, e_temp
            )

        rho_out = wfn_gen_rho(l_wfn_kgrp)
        if symm_rho:
            rho_out.symmetrize()
        rho_out = rho_normalize(rho_out, numel)
        compute_energies()

        e_error = mixmod.compute_error(rho, rho_out)
        if e_error < conv_thr:
            scf_converged = True
        elif idxiter == 0 and e_error < diago_thr * numel:
            diago_thr = 0.1 * e_error / max(1, numel)
            iter_printer(idxiter, scf_converged, e_error, diago_avgiter, en)
            continue
        else:
            diago_thr = min(diago_thr, 0.1 * e_error / max(1, numel))
            diago_thr = max(diago_thr, 1E-13)
            rho = mixmod.mix(rho, rho_out)
            if symm_rho:
                rho.symmetrize()
            rho = rho_normalize(rho, numel)

        rho.Bcast()
        scf_converged = pwcomm.world_comm.bcast(scf_converged)
        diago_thr = pwcomm.world_comm.bcast(diago_thr)
        pwcomm.world_comm.barrier()
        iter_printer(idxiter, scf_converged, e_error, diago_avgiter, en)
        if scf_converged:
            break
        else:
            idxiter += 1

    return scf_converged, rho, l_wfn_kgrp, en
