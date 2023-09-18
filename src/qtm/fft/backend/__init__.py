"""Submodule contaning wrappers to different FFT Libraries.

Supported Libraries are:
 1. MKL-FFT
 2. PyFFTW
 3. SciPy
 4. NumPy
 5. CuPy (GPU)

The interface to the libraries is defined by the `FFTBackend` abstract class.
Refer to its documentation for details on the implementation of the wrappers
in QTM. **You do not need to use the backends defined here for implementing
FFT operations**. FFT methods used across QuantumMASALA are defined in
`qtm.gspace`.

Notes
-----
QuantumMASALA checks if libraries are installed using
``importlib.util.find_spec`` function. Based on the flags, the wrappers are
either imported or skipped to prevent import errors due to missing libraries.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Type
__all__ = [
    'FFTBackend', 'NumPyFFTWrapper', 'SciPyFFTWrapper', 'get_FFTBackend'
]
from .base import FFTBackend
from .numpy_ import NumPyFFTWrapper
from .scipy_ import SciPyFFTWrapper


from importlib.util import find_spec
from qtm.msg_format import value_not_in_list_msg
# NOTE: PyFFTW must be imported ahead of MKL_FFT
# I guess that if mkl_fft is imported first, it might load its FFTW wrapper
# which will be used by PyFFTW. As the mkl wrapper does not support all of
# FFTW's behaviour, it might cause null plans and other issues.


backend_map = {
    'numpy': NumPyFFTWrapper,
    'scipy': SciPyFFTWrapper,
}

if find_spec('pyfftw') is not None:
    from .pyfftw_ import PyFFTWFFTWrapper
    __all__.append('PyFFTWFFTWrapper')
    backend_map['pyfftw'] = PyFFTWFFTWrapper

if find_spec('mkl_fft') is not None:
    from .mklfft_ import MKLFFTWrapper
    __all__.append('MKLFFTWrapper')
    backend_map['mkl_fft'] = MKLFFTWrapper

if find_spec('cupy') is not None:
    from qtm.fft.backend.cupy_ import CuPyFFTWrapper
    __all__.append('CuPyFFTWrapper')
    backend_map['cupy'] = CuPyFFTWrapper


def get_FFTBackend(backend: str | None = None) -> Type[FFTBackend]:
    """Returns the `FFTBackend` type corresponding to input string `backend`

    Parameters
    ----------
    backend : str | None = None, default=None
        Name of the FFT library; Must be one of the elements in the sequence
        `qtm.qtmconfig.fft_available_backends`. If None, the first element
        of the sequence is used.

    Returns
    -------
    Type[FFTBackend]
        `FFTBackend` class corresponding to the library given by `backend`
    """
    from qtm.config import CUPY_INSTALLED, FFT_AVAILABLE_BACKENDS
    if backend == 'cupy':
        if not CUPY_INSTALLED:
            raise RuntimeError(
                "GPU acceleration is disabled. Set 'qtmconfig.use_gpu' to "
                "True to enable GPU routines."
            )

    if backend not in backend_map:
        raise ValueError(
            value_not_in_list_msg('backend', backend, FFT_AVAILABLE_BACKENDS)
        )
    return backend_map[backend]
