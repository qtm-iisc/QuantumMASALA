from dataclasses import dataclass
from typing import Optional, Literal




@dataclass
class PWConfig:
    numkgrp: Optional[int] = None
    pwcomm = None
    use_gpu: bool = False

    fft_type: Literal['slab', 'sticks'] = "slab"
    fft_backend: Optional[Literal['mkl_fft', 'pyfftw',
                                  'scipy', 'numpy']] = 'mkl_fft'
    fft_threads: int = 1
    pyfftw_planner: Literal['FFTW_ESTIMATE', 'FFTW_MEASURE',
                            'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'] = 'FFTW_PATIENT'

    spglib_symprec: float = 1E-5

    libxc_thr_lda_rho: float = 1E-10
    libxc_thr_gga_rho: float = 1E-6
    libxc_thr_gga_sig: float = 1E-10

    eigsolve_method: Literal['davidson', 'primme'] = 'davidson'
    davidson_maxiter: int = 20
    davidson_numwork: int = 4

    mixing_method: Literal['genbroyden', 'mixbroyden',
                           'anderson'] = 'anderson'

    def init_pwcomm(self):
        from .core.pwcomm import PWComm
        self.pwcomm = PWComm(self.numkgrp)


config = PWConfig()
