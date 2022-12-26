"""Contains Abstract Base Classes for FFT Driver and FFT Backends

Written to support different backends (``numpy``, ``scipy``, ``pyfftw``, etc.)
and driver methods (``slab``, ``sticks``)

"""
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
    __slots__ = ['shape', 'ndim', 'n_axes', 'axes']

    def __init__(self, shape, axes):
        self.shape = shape
        """Shape of FFT Arrays (`tuple` of `int`).
        """
        self.ndim = len(self.shape)
        """Dimension of FFT Arrays (`int`).
        """
        self.axes = axes
        """Tuple of axes to perform FFT Along (`tuple` of `int`).
        """
        self.n_axes = len(axes)
        """Number of axes FFT Operations is performed (`int`).
        """

    def do_fft(self, arr):
        """Performs **in-place** FFT on input array. Calls ``self._execute``
        to actually call the backend to perform the operation.

        Parameters
        ----------
        arr : `numpy.ndarray`
            Array to perform FFT, can have any shape as long as the last
            ``self.ndim`` dims have shape ``self.shape``.

        Returns
        -------
        arr : `numpy.ndarray`
            Input array containing the Fourier Transform values present on
            its entry.
        """
        arr_in_ = arr.reshape((-1, *self.shape))
        numfft = arr_in_.shape[0]
        for i in range(numfft):
            self._execute(arr_in_[i], "forward")
        return arr

    def do_ifft(self, arr):
        """Performs **in-place** inverse FFT on input array. Calls
        ``self._execute`` to actually call the backend to perform the operation.

        Parameters
        ----------
        arr : `numpy.ndarray`
            Array to perform inverse FFT, can have any shape as long as the
            last ``self.ndim`` dims have shape ``self.shape``.

        Returns
        -------
        arr : `numpy.ndarray`
            Input array containing the inverse Fourier Transform values present
            on its entry.
        """
        arr_in_ = arr.reshape((-1, *self.shape))
        numfft = arr_in_.shape[0]
        for i in range(numfft):
            self._execute(arr_in_[i], "backward")
        return arr

    @abstractmethod
    def _execute(self, arr, direction):
        """Performs the FFT operation by calling the backend

        Parameters
        ----------
        arr : `numpy.ndarray`
            On entry, contains the input array
        direction : {'forward', 'backward'}
            Direction of FFT to perform
        Returns
        -------
        No object returned. On exit, ``arr`` contains the result of the FFT
        operation
        """
        pass


class FFTModule(ABC):
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

    __slots__ = ["grid_shape", "idxgrid", "numgrid"]

    @abstractmethod
    def __init__(self, grid_shape, idxgrid):
        self.grid_shape = grid_shape
        """Shape of FFT Grid (`tuple` of `int`)
        """
        self.idxgrid = idxgrid
        """Position of G-vectors in the 3D FFT Grid
        (`tuple` [`list` of `int`, ...])
        """
        self.numgrid = len(self.idxgrid[0])
        """Number of G-vectors (`int`)
        """

    def g2r(self, arr_in, arr_out=None, overwrite_in=False):
        """Computes Backwards FFT where input Fourier Transform is 'truncated'

        Parameters
        ----------
        arr_in : `numpy.ndarray`, (..., ``self.numgrid``)
            Input data in G-Space
        arr_out : `numpy.ndarray`, (..., ``*self.grid_shape``), optional
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (..., ``*self.grid_shape``)
            Contains the input data transformed to real-space
        """
        arr_in_ = arr_in.reshape(-1, self.numgrid)

        shape_out = (*arr_in.shape[:-1], *self.grid_shape)
        if arr_out is None:
            arr_out = np.empty(shape_out, dtype="c16")
        elif arr_out.shape != shape_out:
            raise ValueError(f"'arr_out' must be a NumPy array of shape {shape_out}. "
                             f"Got {arr_out.shape}")
        arr_out_ = arr_out.reshape(-1, *self.grid_shape)
        self._g2r(arr_in_, arr_out_, overwrite_in)

        return arr_out

    @abstractmethod
    def _g2r(self, arr_in, arr_out, overwrite_in):
        """Abstract method where the Backwards FFT is perfomed by calling the
        Backend

        Parameters
        ----------
        arr_in : `numpy.ndarray`, (:, ``self.numgrid``)
            Input data in G-Space
        arr_out : `numpy.ndarray`, (:, ``*self.grid_shape``)
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (:, ``*self.grid_shape``)
            Contains the input data transformed to real-space
        """
        pass

    def r2g(self, arr_in, arr_out=None, overwrite_in=False):
        """Computes Forward FFT where output Fourier Transform is 'truncated'

        Parameters
        ----------
        arr_in : `numpy.ndarray`, (..., ``*self.grid_shape``)
            Input data in Real-Space
        arr_out : `numpy.ndarray`, (..., ``self.numgrid``), optional
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (..., ``self.numgrid``)
            Contains the input data transformed to G-space
        """
        arr_in_ = arr_in.reshape((-1, *self.grid_shape))

        shape_out = (*arr_in.shape[:-3], self.numgrid)
        if arr_out is None:
            arr_out = np.empty((*arr_in.shape[:-3], self.numgrid), dtype="c16")
        elif arr_out.shape != shape_out:
            raise ValueError(f"'arr_out' must be a NumPy array of shape {shape_out}. "
                             f"Got {arr_out.shape}")
        arr_out_ = arr_out.reshape(-1, self.numgrid)
        self._r2g(arr_in_, arr_out_, overwrite_in)

        return arr_out

    @abstractmethod
    def _r2g(self, arr_in, arr_out, overwrite_in):
        """Abstract method where the Forward FFT is perfomed by calling the
        Backend

        Parameters
        ----------
        arr_in : `numpy.ndarray`, (:, ``*self.grid_shape``)
            Input data in Real-Space
        arr_out : `numpy.ndarray`, (:, ``self.numgrid``)
            Array to store the output

        Returns
        -------
        arr_out : `numpy.ndarray`, (:, ``self.numgrid``)
            Contains the input data transformed to G-space
        """
        pass
