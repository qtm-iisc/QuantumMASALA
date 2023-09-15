"""QuantumMASALA's Configuration Module

"""
from __future__ import annotations
__all__ = ['QTMConfig', 'qtmconfig', 'NDArray',
           'MPI4PY_INSTALLED', 'MKL_FFT_INSTALLED', 'PYFFTW_INSTALLED',
           'CUPY_INSTALLED', 'FFT_AVAILABLE_BACKENDS']

from typing import Union  # Required import for Runtime checking

import numpy as np
from importlib.util import find_spec

from qtm import logger

MPI4PY_INSTALLED = find_spec('mpi4py') is not None
MKL_FFT_INSTALLED = find_spec('mkl_fft') is not None
PYFFTW_INSTALLED = find_spec('pyfftw') is not None
CUPY_INSTALLED = find_spec('cupy') is not None

FFT_AVAILABLE_BACKENDS = ['scipy', 'numpy']
if PYFFTW_INSTALLED:
    FFT_AVAILABLE_BACKENDS.insert(0, 'pyfftw')
if MKL_FFT_INSTALLED:
    FFT_AVAILABLE_BACKENDS.insert(0, 'mkl_fft')
if CUPY_INSTALLED:
    FFT_AVAILABLE_BACKENDS.append('cupy')

if CUPY_INSTALLED:
    from cupyx import seterr
    seterr(linalg='raise')

fft_use_sticks = False


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
        if not CUPY_INSTALLED:
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

    @property
    def fft_available_backends(self) -> list[str]:
        """list of supported fft libraries installed"""
        return FFT_AVAILABLE_BACKENDS

    _fft_backend = None
    @property  # noqa : E301
    def fft_backend(self) -> str:
        """Configured backend for FFT routines. One of the following:
        ``'mkl_fft'``, ``'pyfftw'``, ``'numpy'``, ``'scipy'``.
        Requires the corresponding library to be installed."""
        if self._fft_backend is None:
            self._fft_backend = FFT_AVAILABLE_BACKENDS[0]
        return self._fft_backend

    @fft_backend.setter
    def fft_backend(self, val):
        if val not in FFT_AVAILABLE_BACKENDS:
            raise ValueError("'fft_backend' must be one of the following: "
                             f"{str(FFT_AVAILABLE_BACKENDS)[1:-1]}. got {val}")
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

    pyfftw_flags: tuple[str, ...] = ()
    """Additional flags to be passed to FFTW Planner routines
    """

    _rng_seed = _RNG_DEFAULT_SEED
    @property  # noqa : E301
    def rng_seed(self):
        """Seed object to be passed to NumPy's RNG routines.
        Automatically broadcasted to all MPI processes."""
        return self._rng_seed

    @rng_seed.setter
    def rng_seed(self, val):
        self._rng_seed = val
        if MPI4PY_INSTALLED:
            from mpi4py.MPI import COMM_WORLD
            self._rng_seed = COMM_WORLD.bcast(self._rng_seed)


qtmconfig: QTMConfig = QTMConfig()
if CUPY_INSTALLED:
    import cupy as cp
    NDArray = Union[np.ndarray, cp.ndarray]
else:
    NDArray = np.ndarray
