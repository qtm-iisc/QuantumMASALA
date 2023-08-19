from __future__ import annotations
__all__ = ['DFTCommMod']
from qtm.mpi.comm import QTMComm, split_comm_pwgrp


class DFTCommMod:

    def __init__(self, image_comm: QTMComm, n_kgrp: int | None = None,
                 pwgrp_size: int = 1):
        self.image_comm = image_comm

        pwgrp_intra, pwgrp_inter = split_comm_pwgrp(self.image_comm, pwgrp_size)
        self.n_pwgrp = pwgrp_inter.size
        self.pwgrp_intra = pwgrp_intra
        self.pwgrp_inter_image = pwgrp_inter

        if n_kgrp is None:
            n_kgrp = self.n_pwgrp
        elif self.n_pwgrp % n_kgrp != 0:
            raise ValueError("'n_kgrp' must evenly divide input 'comm''s"
                             f"{self.n_pwgrp} 'pwgrp' subgroups, but with "
                             f"pwgrp_size = {pwgrp_size} and n_kgrp = {n_kgrp}, "
                             f"it is not possible.")
        self.n_kgrp = n_kgrp

        kgrp_n_pwgrp = self.n_pwgrp // self.n_kgrp
        color = self.pwgrp_inter_image.rank // kgrp_n_pwgrp
        key = self.pwgrp_inter_image.rank % kgrp_n_pwgrp
        self.pwgrp_inter_kgrp = self.pwgrp_inter_image.Split(color, key)
        self.pwgrp_inter_kroot = self.pwgrp_inter_image.Incl(
            tuple(ikgrp * kgrp_n_pwgrp for ikgrp in range(self.n_kgrp))
        )

        kgrp_size = kgrp_n_pwgrp * pwgrp_size
        color, key = self.image_comm.rank // kgrp_size, self.image_comm.rank % kgrp_size
        self.kgrp_intra = self.image_comm.Split(color, key)
        self.kroot_intra = self.image_comm.Incl(
            tuple(ikgrp * kgrp_size for ikgrp in range(self.n_kgrp))
        )


