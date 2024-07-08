from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal
__all__ = ['DFTCommMod', 'DFTConfig']

from qtm.mpi.comm import QTMComm, split_comm_pwgrp
from qtm.logger import qtmlogger
from qtm.msg_format import *


class DFTCommMod:

    def __init__(self, image_comm: QTMComm, n_kgrp: int | None = None,
                 pwgrp_size: int = 1):
        """The DFT Communicator Module

        This image communicaor will be split as follows:
        - n_kgrp nmber of kpoint-pools
        - Each of these will have n_bgrp number of "band groups",
        - Each of these will have "plane-wave-basis groups", of size pwgrp_size
        
        Parameters
        ----------
        image_comm: QTMComm
            Image communicator
        n_kgrp: int
            Number of k-groups/pools to split this image calculation into.
        pwgrp_size: int
            Minimum size of each plane-wave-group.

        """
                
        self.image_comm = image_comm

        pwgrp_intra, pwgrp_inter = split_comm_pwgrp(self.image_comm, pwgrp_size)
        self.n_pwgrp = pwgrp_inter.size     # number of pw-groups
        self.pwgrp_intra = pwgrp_intra      # the pw-group communcator
        self.pwgrp_inter_image = pwgrp_inter  

        if n_kgrp is None:
            n_kgrp = self.n_pwgrp
        elif self.n_pwgrp % n_kgrp != 0:
            raise ValueError("'n_kgrp' must evenly divide input 'comm''s"
                             f"{self.n_pwgrp} 'pwgrp' subgroups, but with "
                             f"pwgrp_size = {pwgrp_size} and n_kgrp = {n_kgrp}, "
                             f"it is not possible.")
        self.n_kgrp = n_kgrp
        self.n_bgrp = self.n_pwgrp // self.n_kgrp

        kgrp_size = self.n_bgrp * pwgrp_size
        self.i_kgrp = self.image_comm.rank // kgrp_size
        key = self.image_comm.rank % kgrp_size
        self.kgrp_intra = self.image_comm.Split(self.i_kgrp, key)
        self.kroot_intra = self.image_comm.Incl(
            tuple(ikgrp * kgrp_size for ikgrp in range(self.n_kgrp))
        )

        self.i_bgrp = self.pwgrp_inter_image.rank % self.n_bgrp
        self.pwgrp_inter_kgrp = self.pwgrp_inter_image.Split(self.i_kgrp, self.i_bgrp)
        self.pwgrp_inter_kroot = self.pwgrp_inter_image.Incl(
            tuple(ikgrp * self.n_bgrp for ikgrp in range(self.n_kgrp))
        )

        qtmlogger.info(f"This image communicator of size {image_comm} has been split as follows:"
                       f"Number of k-pools: {self.n_kgrp}"
                       f"Number of band groups in each k-pool: {self.n_bgrp}"
                       f"Size of each band-group: {pwgrp_size}")



class DFTConfig:

    symm_check_supercell: bool = True
    symm_use_all_frac: bool = False
    spglib_symprec: float = 1E-5

    _eigsolve_method: Literal['davidson', 'scipy'] = 'davidson'
    @property  # noqa : E301
    def eigsolve_method(self):
        return self._eigsolve_method

    @eigsolve_method.setter
    def eigsolve_method(self, val: Literal['davidson', 'scipy']):
        l_solvers = ['davidson', 'scipy']
        if val not in l_solvers:
            raise ValueError(value_not_in_list_msg(
                'DFTConfig.eigsolve_method', val, l_solvers
            ))
        self._eigsolve_method = val

    _davidson_maxiter: int = 20
    @property # noqa : E301
    def davidson_maxiter(self) -> int:
        return self._davidson_maxiter

    @davidson_maxiter.setter
    def davidson_maxiter(self, val: int):
        if not isinstance(val, int) or val <= 0:
            raise TypeError(type_mismatch_msg(
                "DFTConfig.davidson_maxiter", val, 'a positive integer'
            ))
        self._davidson_maxiter = val

    _davidson_numwork: int = 2
    @property  # noqa : E301
    def davidson_numwork(self) -> int:
        return self._davidson_numwork

    @davidson_numwork.setter
    def davidson_numwork(self, val: int):
        if not isinstance(val, int) or val <= 0:
            raise TypeError(type_mismatch_msg(
                "DFTConfig.davidson_numwork", val, 'a positive integer'
            ))

    _mixing_method: Literal['modbroyden'] = 'modbroyden'
    @property  # noqa : E301
    def mixing_method(self):
        return self._mixing_method

    @mixing_method.setter
    def mixing_method(self, val: Literal['modbroyden']):
        l_mixers = ['modbroyden', ]
        if val not in l_mixers:
            raise ValueError(value_not_in_list_msg(
                'DFTConfig.mixing_method', val, l_mixers
            ))

