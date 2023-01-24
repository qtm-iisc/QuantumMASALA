from typing import Type
from .base import FFTDriver, FFTBackend
from .drivers import FFT3D, FFT3DSticks

from quantum_masala import config


def get_fft_driver() -> Type[FFTDriver]:
    if config.fft_use_sticks:
        return FFT3DSticks
    else:
        return FFT3D


__all__ = ["FFT3D", "FFT3DSticks", "get_fft_driver"]
