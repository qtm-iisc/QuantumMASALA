__all__ = ['solve_wfn']

from typing import Any, Callable
from importlib.util import find_spec

from quantum_masala import config
from quantum_masala.core import Wavefun
from quantum_masala.dft.ksham import KSHam

_CUPY_INSTALLED = find_spec('cupy') is not None


def solve_wfn(wfn: Wavefun, gen_ham: Callable[[Wavefun], KSHam],
              diago_thr: float, **prec_params: dict[str, Any]):
    if config.use_gpu:
        if not _CUPY_INSTALLED:
            raise ValueError("'CuPy' not installed. Cannot run in GPU. "
                             "Please set 'USE_GPU' in 'quantum_masala.config' to False"
                             "to run in CPU")
        ham = gen_ham(wfn)
    else:
        ham = gen_ham(wfn)

    if config.eigsolve_method == 'davidson':
        from .davidson import solver
    elif config.eigsolve_method == 'primme':
        from .primme import solver
    else:
        raise ValueError(f"'eigsolve_method' not recognized. Got {config.eigsolve_method}")

    if wfn.noncolin:
        evc_gk, evl = wfn.evc_gk, wfn.evl
        evl[:], evc_gk[:], stat = solver(ham,  diago_thr, evc_gk, **prec_params)
    else:
        for idxspin in range(wfn.numspin):
            ham.set_idxspin(idxspin)
            evc_gk, evl = wfn.evc_gk[idxspin], wfn.evl[idxspin]
            evl[:], evc_gk[:], stat = solver(ham, diago_thr, evc_gk, **prec_params)
            #print(wfn.k_cryst, idxspin, stat)
