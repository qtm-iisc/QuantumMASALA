__all__ = ['solve_wfn']

from typing import Any, Callable
from importlib.util import find_spec

from quantum_masala import config
from quantum_masala.dft.kswfn import KSWavefun
from quantum_masala.dft.ksham import KSHam

_CUPY_INSTALLED = find_spec('cupy') is not None


def solve_wfn(wfn: KSWavefun, gen_ham: Callable[[KSWavefun], KSHam],
              diago_thr: float, **prec_params: dict[str, Any]):
    ham = gen_ham(wfn)

    if config.eigsolve_method == 'davidson':
        from .davidson import solver
    elif config.eigsolve_method == 'primme':
        from .primme import solver
    else:
        raise ValueError(f"'eigsolve_method' not recognized. Got {config.eigsolve_method}")

    if wfn.is_noncolin:
        evc_gk, evl = wfn.evc_gk, wfn.evl
        evl[:], evc_gk[:], stats = solver(ham,  diago_thr, evc_gk, **prec_params)
    else:
        stats = 0
        for idxspin in range(wfn.numspin):
            ham.set_idxspin(idxspin)
            evc_gk, evl = wfn.evc_gk[idxspin], wfn.evl[idxspin]
            evl[:], evc_gk[:], stats_ = solver(ham, diago_thr, evc_gk, **prec_params)
            stats = stats_ + stats
    return stats
