"""QuantumMASALA's Configuration Module

"""
# from __future__ import annotations
from typing import Union, Any, Type
__all__ = ['QTMConfig', 'qtmconfig', 'NDArray']

import numpy as np
from importlib.util import find_spec

from qtm import logger


class QTMConfig:
    """QuantumMASALA's configuration Manager.

    An instance of it is generated at the top-level when the library is loaded
    and is named as `qtmconfig`. Refer below for available config options.
    """

    # Default values
    _RNG_DEFAULT_SEED = 489
    _LOGFILE_DEFAULT_DIR = './qtmpy.log'

    _logging_enabled: bool = False
    @property  # noqa : E301
    def logging_enabled(self) -> bool:
        """True if logging is enabled, else False. Note: Unless `init_logfile` is
        called atleast once, logs are not written to a logfile."""
        return self._logging_enabled

    @logging_enabled.setter
    def logging_enabled(self, val: bool):
        if not isinstance(val, bool):
            raise TypeError("'logging_enabled' must be a boolean. "
                            f"got {val} (type {type(val)}).")
        self._logging_enabled = val
        from logging import disable, NOTSET
        if self.logging_enabled:
            disable(NOTSET)
        else:
            disable()

    logfile_dir: str = _LOGFILE_DEFAULT_DIR
    """Path of the log file"""

    def init_logfile(self):  # noqa : E301
        """Initializes the Logging FileHandler and hooks it to QTM's logger.
        Refer to the `logger` submodule"""
        logger.qtmlogger_set_filehandle(self.logfile_dir)

    @property
    def mpi4py_installed(self) -> bool:
        """True if `mpi4py` is installed, else False"""
        return find_spec('mpi4py') is not None

    @property
    def mkl_fft_installed(self) -> bool:
        """True if `mkl_fft` is installed, else False"""
        return find_spec('mkl_fft') is not None

    @property
    def pyfftw_installed(self) -> bool:
        """True if `pyfftw` is installed, else False"""
        return find_spec('pyfftw') is not None

    @property
    def cupy_installed(self) -> bool:
        """True if `cupy` is installed, else False"""
        return find_spec('cupy') is not None

    def check_cupy(self, suppress_exception: bool = False) -> bool:
        """Checks if CuPy is installed and working

        Returns True if CuPy is installed else False. If CuPy is installed but
        fails to create an array on device, then the function will raise an
        exception if `suppress_exception` is False.

        Parameters
        ----------
        suppress_exception : bool, default=False
            If True, will raise an exception if CuPy is installed but fails to
            allocate a test array on device.

        Returns
        -------
        bool
            True if CuPy is installed

        Raises
        ------
        RuntimeError
            If CuPy is installed, but throws exception(s) when allocating an array
            on device.
        """
        if not self.cupy_installed:
            return False
        try:
            import cupy as cp
            _ = cp.zeros((10, 10, 10), dtype='c16')
            return True
        except Exception as e:
            if suppress_exception:
                return False
            raise RuntimeError(
                "Error encountered when importing cupy and creating a test array. "
                "Refer to exception above for further info."
            ) from e

    _use_gpu: bool = False
    @property  # noqa : E301
    def use_gpu(self) -> bool:
        """enable GPU acceleration"""
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, flag: bool):
        if not isinstance(flag, bool):
            raise TypeError(f"'use_gpu' must be a boolean. got type {type(flag)}")
        self._use_gpu = False
        if flag:
            self.check_cupy()
            self._use_gpu = True

    @property
    def NDArray(self) -> Type[np.ndarray]:  # noqa : N802
        """`typing.Union[numpy.ndarray, cupy.ndarray]` if `use_gpu` is True,
        else `numpy.ndarray`. Primarily for typing routines that are
        CPU(NumPy)/GPU(CuPy) Agnostic.
        """
        if self.cupy_installed:
            import cupy as cp
            ndarray = Union[np.ndarray, cp.ndarray]
        else:
            ndarray = np.ndarray
        return ndarray

    @property
    def fft_available_backends(self) -> list[str]:
        """list of supported fft libraries installed"""
        backends = ['scipy', 'numpy']
        if self.pyfftw_installed:
            backends.insert(0, 'pyfftw')
        if self.mkl_fft_installed:
            backends.insert(0, 'mkl_fft')
        return backends

    _fft_backend = None
    @property  # noqa : E301
    def fft_backend(self) -> str:
        """Configured backend for FFT routines. One of the following:
        ``'mkl_fft'``, ``'pyfftw'``, ``'numpy'``, ``'scipy'``.
        Requires the corresponding library to be installed."""
        if self._fft_backend is None:
            self._fft_backend = self.fft_available_backends[0]
        return self._fft_backend

    @fft_backend.setter
    def fft_backend(self, val):
        if val not in self.fft_available_backends:
            raise ValueError("'fft_backend' must be one of the following: "
                             f"{str(self.fft_available_backends)[1:-1]}. got {val}")
        self._fft_backend = val

    _fft_threads: int = 1
    @property
    def fft_threads(self) -> int:
        """Number of threads used in FFT routines. Supported only when `backend` is
        ``'pyfftw'`` or ``'scipy'``"""
        return self._fft_threads

    @fft_threads.setter
    def fft_threads(self, val: int):
        if not isinstance(val, int) or val < 1:
            raise ValueError("'fft_threads' must be a positive integer. "
                             f"got {val} (type {type(val)})")
        self._fft_threads = val

    _pyfftw_planner = None
    @property  # noqa : E301
    def pyfftw_planner(self) -> str:
        """FFTW Planner flags. Effective only when `fft_backend` is `pyfftw`.
        One of the following: ``'FFTW_ESTIMATE'``, ``'FFTW_MEASURE'`` (default),
        ``'FFTW_PATIENT'``, ``'FFTW_EXHAUSTIVE'``"""
        if self._pyfftw_planner is None:
            self._pyfftw_planner = 'FFTW_MEASURE'
        return self._pyfftw_planner

    @pyfftw_planner.setter
    def pyfftw_planner(self, val):
        available_planners = (
            'FFTW_ESTIMATE', 'FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'
        )
        if val not in available_planners:
            raise ValueError("'pyfftw_planner' must be one of the followinig: "
                             f"{str(available_planners)[1:-1]}. got {val}")
        self._pyfftw_planner = val

    pyfftw_flags: tuple[str, ...] = ( )
    """Additional flags to be passed to FFTW Planner routines
    """

    _rng_seed: Any = _RNG_DEFAULT_SEED
    @property  # noqa : E301
    def rng_seed(self) -> Any:
        """Seed object to be passed to NumPy's RNG routines.
        Automatically broadcasted to all MPI processes."""
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, val):
        self._rng_seed = val
        if self.mpi4py_installed:
            from mpi4py.MPI import COMM_WORLD
            self._rng_seed = COMM_WORLD.bcast(self._rng_seed)


qtmconfig = QTMConfig()
NDArray = qtmconfig.NDArray
