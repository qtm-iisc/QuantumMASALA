from abc import ABC, abstractmethod

from quantum_masala.core import PWComm, GSpace, Rho, FFTGSpace


class LocalPot(ABC):
    """Abstract Class for local potentials.

    All local potentials Subclass this class

    Attributes
    ----------
    pwcomm : PWComm
        Module for communication between MPI processes in parallel run
    grho : GSpace
        Represents the G-Space of 'hard grid'; same as in charge density
    fft_rho : FFTGSpace
        FFT Interface of
    """

    @abstractmethod
    def __init__(self, rho: Rho):
        self.rho = rho
        self.pwcomm = self.rho.pwcomm
        self.grho = self.rho.grho
        self.fft_rho = self.rho.fft_rho

    @property
    @abstractmethod
    def r(self):
        pass

    @property
    @abstractmethod
    def g(self):
        pass

    @property
    @abstractmethod
    def en(self):
        pass

    @abstractmethod
    def sync(self):
        pass

    @abstractmethod
    def compute(self):
        pass
