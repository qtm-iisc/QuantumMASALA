from importlib.util import find_spec

from .npfft import NpFFTLibWrapper
from .spfft import SpFFTLibWrapper

__all__ = ["NpFFTLibWrapper", "SpFFTLibWrapper"]

if find_spec('pyfftw'):
    from .pyfftw import PyFFTWLibWrapper
    __all__.append("PyFFTWLibWrapper")

if find_spec('mklfft'):
    from .mklfft import MKLFFTLibWrapper
    __all__.append("MKLFFTLibWrapper")
