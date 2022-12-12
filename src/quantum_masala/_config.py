# TODO: Refactor this where fft_backend is a property with setters
from os import getenv
from dataclasses import dataclass
from typing import Optional, Literal




@dataclass
class PWConfig:
    numkgrp: Optional[int] = None
    pwcomm = None
    use_gpu: bool = False

    fft_type: Literal['slab', 'sticks'] = "slab"
    fft_backend: Optional[Literal['mkl_fft', 'pyfftw',
                                  'scipy', 'numpy']] = None
    fft_threads: int = int(getenv("OMP_NUM_THREADS", "1"))
    pyfftw_planner: Literal['FFTW_ESTIMATE', 'FFTW_MEASURE',
                            'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'] = 'FFTW_MEASURE'

    spglib_symprec: float = 1E-5

    libxc_thr_lda_rho: float = 1E-10
    libxc_thr_gga_rho: float = 1E-6
    libxc_thr_gga_sig: float = 1E-10

    eigsolve_method: Literal['davidson', 'primme'] = 'davidson'
    davidson_maxiter: int = 20
    davidson_numwork: int = 2

    mixing_method: Literal['genbroyden', 'mixbroyden',
                           'anderson'] = 'anderson'

    tddft_exp_method: Literal['taylor', 'splitoperator'] = 'taylor'
    taylor_order: int = 4
    tddft_prop_method: Literal['etrs', 'splitoperator'] = 'etrs'

    def init_pwcomm(self):
        from .core.pwcomm import PWComm
        self.pwcomm = PWComm(self.numkgrp)


config = PWConfig()
