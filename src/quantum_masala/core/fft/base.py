"""Contains Abstract Base Classes for FFT Driver and FFT Backends

Written to support different backends (``numpy``, ``scipy``, ``pyfftw``, etc.)
and driver methods (``slab``, ``sticks``)

"""
from typing import Any, Optional
from abc import ABC, abstractmethod

import numpy as np


class FFTBackend(ABC):
    """Abstract Base Class for interfacing with different FFT Backends

    The shape of the array to perform FFT is assumed to be fixed for each
    instance when initializing, allowing generation of plans at startup.

    Parameters
    ----------
    shape : `tuple` of `int`
        Shape of the array to perform FFT
    axes : `tuple` of `int
        Tuple of axes to perform FFT
    """
    __slots__ = ['shape', 'strides', 'axes', 'normalise_idft',
                 'plan_fw', 'plan_bw']

    def __init__(self, arr: np.ndarray, axes: tuple[int, ...], normalise_idft: bool):
        self.shape: tuple[int, ...] = arr.shape
        """Shape of FFT Arrays.
        """
        self.strides: tuple[int, ...] = arr.strides
        """Dimension of FFT Arrays.
        """
        self.axes: tuple[int, ...] = axes
        """Tuple of axes to perform FFT Along.
        """
        self.normalise_idft: bool = normalise_idft
        """`True` if Inverse DFT is scaled by 1/N where N is 
        the product of axis lengths
        """
        self.plan_fw: Any = None
        self.plan_bw: Any = None

    @classmethod
    @abstractmethod
    def create_buffer(cls, shape: tuple[int, ...]):
        pass

    @abstractmethod
    def fft(self, arr: np.ndarray):
        pass

    @abstractmethod
    def ifft(self, arr: np.ndarray):
        pass


class FFTDriver(ABC):
    """Class Specification for FFT Drivers

    FFT Drivers must inherit this Abstract Base Class and implement required
    methods to ensure compatibility with implemented FFT Interfaces

    Parameters
    ----------
    grid_shape : `tuple` of `int`
        Shape of the 3D FFT Grid
    idxgrid : `tuple` [`list` of `int`, ...]
        List of indices that are within the truncated Fourier Space
    """

    __slots__ = ["grid_shape", "idxgrid", "numgrid", "normalise_idft"]

    @abstractmethod
    def __init__(self, grid_shape: tuple[int, int, int],
                 idxgrid: tuple[list[int], ...],
                 normalise_idft: bool = True):
        self.grid_shape: tuple[int, int, int] = grid_shape
        """Shape of FFT Grid.
        """
        self.idxgrid: tuple[list[int], ...] = idxgrid
        """Position of G-vectors in the 3D FFT Grid.
        """
        self.numgrid: int = len(self.idxgrid[0])
        """Number of G-vectors.
        """
        self.normalise_idft: bool = normalise_idft
        """`True` if Inverse DFT is scaled by 1/N where N is 
        the product of axis lengths
        """

    def g2r(self, arr_inp: np.ndarray, arr_out: Optional[np.ndarray] = None):
        """Computes Backwards FFT where input Fourier Transform is 'truncated'

        Parameters
        ----------
        arr_inp : `numpy.ndarray`, (..., ``self.numgrid``)
            Input data in G-Space
        arr_out : `numpy.ndarray`, (..., ``*self.grid_shape``), optional
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (..., ``*self.grid_shape``)
            Contains the input data transformed to real-space
        """
        arr_inp_ = arr_inp.reshape(-1, self.numgrid)

        shape_out = (*arr_inp.shape[:-1], *self.grid_shape)
        if arr_out is None:
            arr_out = np.empty(shape_out, dtype="c16")
        elif arr_out.shape != shape_out:
            raise ValueError(f"'arr_out' must be a NumPy array of shape {shape_out}. "
                             f"Got {arr_out.shape}")
        arr_out_ = arr_out.reshape(-1, *self.grid_shape)
        for inp, out in zip(arr_inp_, arr_out_):
            self._g2r(inp, out)

        return arr_out

    @abstractmethod
    def _g2r(self, arr_inp: np.ndarray, arr_out: np.ndarray):
        """Abstract method where the Backwards FFT is perfomed by calling the
        Backend

        Parameters
        ----------
        arr_inp : `numpy.ndarray`, (:, ``self.numgrid``)
            Input data in G-Space
        arr_out : `numpy.ndarray`, (:, ``*self.grid_shape``)
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (:, ``*self.grid_shape``)
            Contains the input data transformed to real-space
        """
        pass

    def r2g(self, arr_inp: np.ndarray, arr_out: Optional[np.ndarray] = None):
        """Computes Forward FFT where output Fourier Transform is 'truncated'

        Parameters
        ----------
        arr_inp : `numpy.ndarray`, (..., ``*self.grid_shape``)
            Input data in Real-Space
        arr_out : `numpy.ndarray`, (..., ``self.numgrid``), optional
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (..., ``self.numgrid``)
            Contains the input data transformed to G-space
        """
        arr_inp_ = arr_inp.reshape((-1, *self.grid_shape))

        shape_out = (*arr_inp.shape[:-3], self.numgrid)
        if arr_out is None:
            arr_out = np.empty((*arr_inp.shape[:-3], self.numgrid), dtype="c16")
        elif arr_out.shape != shape_out:
            raise ValueError(f"'arr_out' must be a NumPy array of shape {shape_out}. "
                             f"Got {arr_out.shape}")
        arr_out_ = arr_out.reshape(-1, self.numgrid)
        for inp, out in zip(arr_inp_, arr_out_):
            self._r2g(inp, out)

        return arr_out

    @abstractmethod
    def _r2g(self, arr_inp: np.ndarray, arr_out: np.ndarray):
        """Abstract method where the Forward FFT is perfomed by calling the
        Backend

        Parameters
        ----------
        arr_inp : `numpy.ndarray`, (:, ``*self.grid_shape``)
            Input data in Real-Space
        arr_out : `numpy.ndarray`, (:, ``self.numgrid``)
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (:, ``self.numgrid``)
            Contains the input data transformed to G-space
        """
        pass
