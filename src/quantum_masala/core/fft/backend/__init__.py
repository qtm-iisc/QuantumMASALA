from importlib.util import find_spec

from .numpy_ import NpFFTLibWrapper
from .scipy_ import SpFFTLibWrapper
__all__ = ["get_fft_backend", "NpFFTLibWrapper", "SpFFTLibWrapper"]

_PYFFTW_INSTALLED = find_spec("pyfftw") is not None
if _PYFFTW_INSTALLED:
    from .pyfftw_ import PyFFTWLibWrapper
    __all__.append("PyFFTWLibWrapper")

_MKLFFT_INSTALLED = find_spec("mkl_fft") is not None
if _MKLFFT_INSTALLED:
    from .mklfft_ import MKLFFTLibWrapper
    __all__.append("MKLFFTLibWrapper")


def get_fft_backend():
    from quantum_masala import config
    if config.fft_backend is None:
        return eval(__all__[-1])  # TODO: Replace this bodge
    elif config.fft_backend == "mkl_fft":
        if _MKLFFT_INSTALLED:
            return MKLFFTLibWrapper
        raise ValueError("'mkl_fft' not installed. Choose a different backend"
                         f"by setting a different 'fft_backend'. Got '{config.fft_backend}'")
    elif config.fft_backend == "pyfftw":
        if _PYFFTW_INSTALLED:
            return PyFFTWLibWrapper
        raise ValueError("'pyfftw' not installed. Choose a different backend"
                         f"by setting a different 'fft_backend'. Got '{config.fft_backend}'")
    elif config.fft_backend == "scipy":
        return SpFFTLibWrapper
    elif config.fft_backend == "numpy":
        return NpFFTLibWrapper
    else:
        raise ValueError(f"'fft_backend' not recognized. Got '{config.fft_backend}'")