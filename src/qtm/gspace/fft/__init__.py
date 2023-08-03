from qtm import config
from .base import FFT3D#, FFTDriver
from .full import FFT3DFull
from .sticks import FFT3DSticks
from typing import Type

from . import utils

def get_fft_driver() -> Type[FFT3D]:
    if config.fft_use_sticks:
        return FFT3DSticks
    else:
        return FFT3DFull