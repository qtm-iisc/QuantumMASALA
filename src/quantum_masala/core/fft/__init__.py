from .base import FFTModule, FFTBackend
from .fftslab import FFTSlab
from .fftstick import FFTStick

from quantum_masala import config


def get_fft_module():
    if config.fft_type == 'slab':
        return FFTSlab
    elif config.fft_type == 'stick':
        return FFTStick
    else:
        raise ValueError(f"'fft_type' not recognized. Got {config.fft_type}")


__all__ = ["FFTSlab", "FFTStick", "get_fft_module"]
