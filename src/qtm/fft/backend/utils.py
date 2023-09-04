
from typing import Type
from qtm import config

from qtm.fft.base import FFT3D
from qtm.fft.full import FFT3DFull
from qtm.fft.sticks import FFT3DSticks


def get_fft_driver() -> Type[FFT3D]:
    if config.fft_use_sticks:
        return FFT3DSticks
    else:
        return FFT3DFull