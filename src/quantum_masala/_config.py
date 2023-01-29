# TODO: Refactor this where fft_backend is a property with setters
from os import getenv
from importlib.util import find_spec

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


@dataclass
class PWConfig:
    _numkgrp: int = None
    _use_gpu: bool = False
    _pwcomm = None

    fft_use_sticks: bool = False
    fft_backend: Optional[Literal['mkl_fft', 'pyfftw',
                                  'scipy', 'numpy']] = 'pyfftw'
    fft_threads: int = int(getenv("OMP_NUM_THREADS", "1"))
    pyfftw_planner: Literal['FFTW_ESTIMATE', 'FFTW_MEASURE',
                            'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'] = 'FFTW_MEASURE'
    pyfftw_flags: tuple[str, ...] = ('FFTW_DESTROY_INPUT', )

    symm_check_supercell: bool = True
    symm_use_all_frac: bool = False
    spglib_symprec: float = 1E-5

    # libxc_thr_lda_rho: float = 1E-10
    # libxc_thr_gga_rho: float = 1E-6
    # libxc_thr_gga_sig: float = 1E-10

    eigsolve_method: Literal['davidson', 'primme'] = 'davidson'
    davidson_maxiter: int = 20
    davidson_numwork: int = 2

    mixing_method: Literal['genbroyden', 'modbroyden',
                           'anderson'] = 'modbroyden'

    tddft_exp_method: Literal['taylor', 'splitoper'] = 'taylor'
    taylor_order: int = 4
    tddft_prop_method: Literal['etrs', 'splitoperator'] = 'etrs'

    logfile: bool = False
    logfile_name: Optional[str] = 'QTMPy.log'

    _rng_seed: int = 489
    _rng: np.random.Generator = np.random.default_rng(_rng_seed)

    @property
    def rng_seed(self):
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, seed):
        self._rng_seed = seed
        self._rng = np.random.default_rng(self._rng_seed)

    @property
    def rng(self):
        return self._rng

    @property
    def numkgrp(self):
        return self._numkgrp

    @numkgrp.setter
    def numkgrp(self, numkgrp: Optional[int]):
        numproc = 1
        if find_spec("mpi4py") is not None:
            from mpi4py.MPI import COMM_WORLD
            numproc = COMM_WORLD.Get_size()

        if numkgrp is None:
            numkgrp = numproc
        elif not isinstance(numkgrp, int) or numkgrp < 1:
            raise ValueError("'numkgrp' must be a positive integer. "
                             f"Got {numkgrp} (type '{type(numkgrp)}')")
        if numkgrp > numproc or numproc % numkgrp != 0:
            raise ValueError(
                "'numkgrp' must divide the number of MPI Processes evenly. "
                f"Got 'numproc={numproc}', 'numkgrp={numkgrp}'"
            )
        self._numkgrp = numkgrp

        from .core.pwcomm import PWComm
        self._pwcomm = PWComm(self._numkgrp)

        from quantum_masala import pw_logger
        from quantum_masala.logger import logger_set_filehandle
        if self.logfile:
            logger_set_filehandle(self.logfile_name)

        pw_logger.log_message('world_rank/world_size - kgrp_rank/kgrp_size: '
                              f'{self.pwcomm.world_rank}/{self._pwcomm.world_size} - '
                              f'{self.pwcomm.kgrp_rank}/{self._pwcomm.kgrp_size}')

    @property
    def use_gpu(self):
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, use_gpu: bool):
        if not isinstance(use_gpu, bool):
            raise TypeError(f"'use_gpu' must be a boolean. Got '{type(use_gpu)}'")
        if use_gpu and find_spec('cupy') is None:
            raise ModuleNotFoundError(
                "'CuPy' cannot be located. 'use_gpu' can be set to 'True' only "
                "when 'CuPy' is installed."
            )
        self._use_gpu = use_gpu

    @property
    def pwcomm(self):
        if self._numkgrp is None:
            from quantum_masala import pw_logger
            from quantum_masala.logger import logger_set_filehandle
            if self.logfile:
                logger_set_filehandle(self.logfile_name)

            numproc = 1
            if find_spec("mpi4py") is not None:
                from mpi4py.MPI import COMM_WORLD
                numproc = COMM_WORLD.Get_size()

            pw_logger.warn("'numkgrp' has not bee initialized. Setting it to "
                           f"{numproc} (# of processes in COMM_WORLD).\n"
                           "If you want to enable band distribution, please specify "
                           "the appropriate # of k-groups by setting "
                           "'quantum_masala.config.numkgrp'")

        return self._pwcomm


config = PWConfig()
