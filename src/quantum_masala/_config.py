from os import getenv
from argparse import ArgumentParser
from importlib.util import find_spec

from dataclasses import dataclass
from typing import Optional, Literal

RNG_DEFAULT_SEED = 489
LOGFILE_DEFAULT_DIR = './qtmpy.log'

argparser = ArgumentParser()
argparser.add_argument('-nkgrp', '-nk', type=int, default=None,
                       help='number of k groups to split MPI processes across')
argparser.add_argument('--use-gpu', action='store_true',
                       help="enable gpu acceleration (if applicable). "
                            "requires 'cupy' library to be installed")
argparser.add_argument('-log', action='store_true',
                       help='enable event logging to file')
argparser.add_argument('--logfile', action='store', type=str,
                       help=f'name of the logfile. '
                            f'default is {LOGFILE_DEFAULT_DIR}')


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
    logfile_name: str = LOGFILE_DEFAULT_DIR

    _rng_seed: int = RNG_DEFAULT_SEED

    @property
    def rng_seed(self):
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, seed):
        self._rng_seed = seed
        if find_spec("mpi4py") is not None:
            from mpi4py.MPI import COMM_WORLD
            self._rng_seed = COMM_WORLD.bcast(self._rng_seed)

    @property
    def numkgrp(self):
        return self._numkgrp

    @numkgrp.setter
    def numkgrp(self, numkgrp: int):
        print('setting numkgrp: ', numkgrp)
        numproc = 1
        if find_spec("mpi4py") is not None:
            from mpi4py.MPI import COMM_WORLD
            numproc = COMM_WORLD.Get_size()

        if not isinstance(numkgrp, int) or numkgrp < 1:
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

        from .logger import pw_logger
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
        if use_gpu:
            try:
                import cupy as cp
                _ = cp.zeros((10, 10), dtype='c16')
            except Exception as e:
                raise RuntimeError(
                    "Error encountered when importing cupy and creating a test array. "
                    "Refer to exception above for further info."
                ) from e
        self._use_gpu = use_gpu

    @property
    def pwcomm(self):
        return self._pwcomm

    def parse_args(self):
        args = argparser.parse_args()
        print(args)
        self.use_gpu = args.use_gpu
        self.logfile = args.log or args.logfile is not None
        if self.logfile:
            if args.logfile is None:
                self.logfile_name = LOGFILE_DEFAULT_DIR
            else:
                self.logfile_name = args.logfile

        numkgrp = args.nkgrp
        if numkgrp is None:
            from .logger import pw_logger, logger_set_filehandle
            if self.logfile:
                logger_set_filehandle(self.logfile_name)

            numproc = 1
            if find_spec("mpi4py") is not None:
                from mpi4py.MPI import COMM_WORLD
                numproc = COMM_WORLD.Get_size()

            if numproc != 1:
                pw_logger.warn(
                    f"'numkgrp' has not been specified. Setting it to {numproc} "
                    "(# of processes in COMM_WORLD).\n"
                    "To enable band distribution, please specify the appropriate "
                    "number of k-groups using command line option '-nk'/'-nkgrp' "
                    "or setting 'quantum_masala.config.numkgrp'"
                )
            numkgrp = numproc
        self.numkgrp = numkgrp


config = PWConfig()
