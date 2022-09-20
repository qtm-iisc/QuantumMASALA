from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np


class FFTLibWrapper(ABC):
    __slots__ = ['shape', 'ndim', 'n_axes', 'axes']

    def __init__(self, shape: tuple[int, ...], axes: Union[int, tuple[int, ...]]):
        self.shape = shape
        self.ndim = len(self.shape)

        if isinstance(axes, int):
            self.n_axes = axes
            self.axes = tuple([-i - 1 for i in range(self.n_axes)])
        elif isinstance(axes, tuple):
            self.axes = axes
            self.n_axes = len(axes)

    def do_fft(self, arr_in: np.ndarray, arr_out: np.ndarray) -> np.ndarray:
        n_arr = arr_in.reshape((-1,) + self.shape).shape[0]
        arr_in_ = arr_in.reshape((-1, *self.shape))
        arr_out_ = arr_out.reshape((-1, *self.shape))
        for i in range(n_arr):
            self._execute(arr_in_[i], arr_out_[i], "forward")

        return arr_out

    def do_ifft(self, arr_in: np.ndarray, arr_out: np.ndarray) -> np.ndarray:
        n_arr = arr_in.reshape((-1,) + self.shape).shape[0]
        arr_in_ = arr_in.reshape((-1, *self.shape))
        arr_out_ = arr_out.reshape((-1, *self.shape))
        for i in range(n_arr):
            self._execute(arr_in_[i], arr_out_[i], "backward")

        return arr_out

    @abstractmethod
    def _execute(
        self,
        arr_in: np.ndarray,
        arr_out: np.ndarray,
        direction: Literal["forward", "backward"],
    ) -> np.ndarray:
        pass
