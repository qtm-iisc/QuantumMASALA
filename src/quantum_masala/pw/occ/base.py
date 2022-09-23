from abc import ABC, abstractmethod
from typing import Type
import numpy as np

from quantum_masala.core import PWComm, WfnK


class OccMod(ABC):

    pwcomm: PWComm = PWComm()

    @abstractmethod
    def __init__(self, WfnK_: Type[WfnK], numel: float):
        self.WfnK = WfnK_

        self.numspin = self.WfnK.numspin
        self.numbnd = self.WfnK.numbnd
        self.bnd_par = self.WfnK.bnd_par

        self.numel = numel

        self.e_smear = None
        self.e_mu = 0.0

    def _compute_numel(self, occ: np.ndarray, weights: np.ndarray):
        fac = 2 if self.numspin == 1 else 1

        numel_proc = fac * np.sum(occ * weights.reshape((-1, 1, 1)))
        numel = None
        if self.bnd_par:
            numel = self.pwcomm.world_comm.allreduce_scalar_sum(numel_proc)
        else:
            if self.pwcomm.is_kgrp_root:
                numel = self.pwcomm.kgrp_intercomm.allreduce_sum(numel_proc)
            numel = self.pwcomm.world_comm.bcast(numel)
        return numel

    @abstractmethod
    def _compute_en(self, occ: np.ndarray, e_mu, weights: np.ndarray):
        pass

    @abstractmethod
    def compute(self, l_wfnk: list[WfnK]):
        pass
