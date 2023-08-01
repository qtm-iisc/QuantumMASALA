"""Submodule contaning wrappers to different FFT Libraries.

Supported Libraries are:
1. MKL-FFT
2. PyFFTW
3. SciPy
4. NumPy
5. CuPy (GPU)

The abstract class `FFTBackend` that is defined in ``../base.py`` describes
the structure of the wrappers definend here.

Notes
-----
QuantumMASALA checks if libraries are installed using
``importlib.util.find_spec`` function, and sets the ``*_INSTALLED`` flags
in `qtm.config`. Based on the flags, the wrappers are either imported
or skipped to prevent import errors due to missing libraries.
"""

from .numpy_ import NumPyFFTWrapper
from .scipy_ import SciPyFFTWrapper
__all__ = ['NumPyFFTWrapper', 'SciPyFFTWrapper', 'get_FFTBackend']

from qtm import qtmconfig
# NOTE: PyFFTW must be imported ahead of MKL_FFT
if qtmconfig.pyfftw_installed:
    from .pyfftw_ import PyFFTWFFTWrapper
    __all__.append('PyFFTWFFTWrapper')
if qtmconfig.mkl_fft_installed:
    from .mklfft_ import MKLFFTWrapper
    __all__.append('MKLFFTWrapper')
if qtmconfig.cupy_installed:
    if qtmconfig.check_cupy(suppress_exception=True):
        from qtm.gspace.fft.backend.cupy_ import CuPyFFTWrapper
        __all__.append('CuPyFFTWrapper')


def get_FFTBackend(backend: str):
    if backend == 'cupy':
        if not qtmconfig.use_gpu:
            raise ValueError("GPU acceleration is disabled. "
                             "Set 'qtmconfig.use_gpu' to True to enable GPU routines.")
        else:
            return CuPyFFTWrapper

    available_backends = qtmconfig.fft_available_backends
    if qtmconfig.use_gpu:
        available_backends.append('cupy')

    if backend not in available_backends:
        raise ValueError("'backend' must be one of the following: "
                         f"{str(available_backends)[1:-1]}. got {backend}")
    if backend == 'pyfftw':
        FFTBackend_ = PyFFTWFFTWrapper
    elif backend == 'mkl_fft':
        FFTBackend_ = MKLFFTWrapper
    elif backend == 'scipy':
        FFTBackend_ = SciPyFFTWrapper
    else:
        FFTBackend_ = NumPyFFTWrapper
    return FFTBackend_
