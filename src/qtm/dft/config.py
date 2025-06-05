from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
__all__ = ["DFTCommMod", "DFTConfig"]

from qtm.config import MPI4PY_INSTALLED, PRIMME_INSTALLED
from qtm.logger import qtmlogger
from qtm.mpi.comm import QTMComm, split_comm_pwgrp
from qtm.msg_format import *

class DFTCommMod:
    def __init__(
        self, image_comm: QTMComm, n_kgrp: int | None = None, pwgrp_size: int = 1
    ):
        """
        Initializes the DFTCommMod object.

        The image-comm is split uniformly into pwgrps, each with size 'pwgrp_size'.
        Each plane-wave-group (pwgrp) is allotted to a band-group (bgrp),
        and each bgrp is allotted to one of the `n_kgrp` k-groups (kgrp).

        Args:
            image_comm (QTMComm): The communication object.
            n_kgrp (int | None, optional): The number of k-groups. Defaults to None (interpreted as image_comm.size//pwgrp_size).
            pwgrp_size (int, optional): The size of each pw-group. Defaults to 1.

        Raises:
            ValueError: If 'n_kgrp' does not evenly divide the number of 'pwgrp' subgroups.
        """
        self.image_comm = image_comm
        if MPI4PY_INSTALLED:
            pwgrp_intra, pwgrp_inter = split_comm_pwgrp(self.image_comm, pwgrp_size)
            self.n_pwgrp = pwgrp_inter.size
            self.pwgrp_intra = pwgrp_intra
            """Intra-pwgrp communicator, with size = pwgrp_size."""
            self.pwgrp_inter_image = pwgrp_inter
            """Inter-pwgrp communicator, with size = n_procs // pwgrp_size."""

            if n_kgrp is None:
                n_kgrp = self.n_pwgrp
            elif self.n_pwgrp % n_kgrp != 0:
                raise ValueError(
                    "'n_kgrp' must evenly divide input 'comm''s"
                    f"{self.n_pwgrp} 'pwgrp' subgroups, but with "
                    f"pwgrp_size = {pwgrp_size} and n_kgrp = {n_kgrp}, "
                    f"it is not possible."
                )
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
            """Index of the band-group (bgrp) within it's k-group."""
            self.pwgrp_inter_kgrp = self.pwgrp_inter_image.Split(
                self.i_kgrp, self.i_bgrp
            )
            self.pwgrp_inter_kroot = self.pwgrp_inter_image.Incl(
                tuple(ikgrp * self.n_bgrp for ikgrp in range(self.n_kgrp))
            )
        else:
            self.n_pwgrp = 1
            self.pwgrp_intra = None
            self.pwgrp_inter_image = self.image_comm
            self.n_kgrp = 1
            self.n_bgrp = 1
            self.i_kgrp = 0
            self.kgrp_intra = self.image_comm
            self.kroot_intra = self.image_comm
            self.i_bgrp = 0
            self.pwgrp_inter_kgrp = self.image_comm
            self.pwgrp_inter_kroot = self.image_comm

    def __repr__(self) -> str:
        return (
            f"DFTCommMod(image_comm.size={self.image_comm.size}, n_kgrp={self.n_kgrp}, "
            f"n_bgrp={self.n_bgrp}, pwgrp_size={self.pwgrp_intra.size if self.pwgrp_intra is not None else None})"
        )


class DFTConfig:
    symm_check_supercell: bool = True
    symm_use_all_frac: bool = False
    spglib_symprec: float = 1e-5

    _eigsolve_method: Literal["davidson", "scipy", "primme"] = "davidson"

    @property  # noqa : E301
    def eigsolve_method(self):
        return self._eigsolve_method

    @eigsolve_method.setter
    def eigsolve_method(self, val: Literal["davidson", "scipy", "primme"]):
        l_solvers = ["davidson", "scipy"]
        if PRIMME_INSTALLED:
            l_solvers.append("primme")
        elif val == "primme":
            raise ImportError("Primme is not installed")
        if val not in l_solvers:
            raise ValueError(
                value_not_in_list_msg("DFTConfig.eigsolve_method", val, l_solvers)
            )
        self._eigsolve_method = val

    _davidson_maxiter: int = 20

    @property  # noqa : E301
    def davidson_maxiter(self) -> int:
        return self._davidson_maxiter

    @davidson_maxiter.setter
    def davidson_maxiter(self, val: int):
        if not isinstance(val, int) or val <= 0:
            raise TypeError(
                type_mismatch_msg(
                    "DFTConfig.davidson_maxiter", val, "a positive integer"
                )
            )
        self._davidson_maxiter = val

    _davidson_numwork: int = 2

    @property  # noqa : E301
    def davidson_numwork(self) -> int:
        return self._davidson_numwork

    @davidson_numwork.setter
    def davidson_numwork(self, val: int):
        if not isinstance(val, int) or val <= 0:
            raise TypeError(
                type_mismatch_msg(
                    "DFTConfig.davidson_numwork", val, "a positive integer"
                )
            )

    _mixing_method: Literal["modbroyden", "genbroyden"] = "modbroyden"

    @property  # noqa : E301
    def mixing_method(self):
        return self._mixing_method

    @mixing_method.setter
    def mixing_method(self, val: Literal["modbroyden", "genbroyden"]):
        l_mixers = ["modbroyden", "genbroyden"]
        if val not in l_mixers:
            raise ValueError(
                value_not_in_list_msg("DFTConfig.mixing_method", val, l_mixers)
            )

    def __repr__(self) -> str:
        return (
            f"DFTConfig(symm_check_supercell={self.symm_check_supercell}, "
            f"symm_use_all_frac={self.symm_use_all_frac}, "
            f"spglib_symprec={self.spglib_symprec}, "
            f"eigsolve_method={self.eigsolve_method}, "
            f"davidson_maxiter={self.davidson_maxiter}, "
            f"davidson_numwork={self.davidson_numwork}, "
            f"mixing_method={self.mixing_method})"
        )

    def __str__(self, indent=" " * 22) -> str:
        return (
            f"DFTConfig(\n"
            f"{indent}symm_check_supercell = {self.symm_check_supercell},\n"
            f"{indent}symm_use_all_frac = {self.symm_use_all_frac},\n"
            f"{indent}spglib_symprec = {self.spglib_symprec},\n"
            f"{indent}eigsolve_method = '{self.eigsolve_method}',\n"
            f"{indent}davidson_maxiter = {self.davidson_maxiter},\n"
            f"{indent}davidson_numwork = {self.davidson_numwork},\n"
            f"{indent}mixing_method = '{self.mixing_method}')"
        )
