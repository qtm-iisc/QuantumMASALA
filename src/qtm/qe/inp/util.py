from typing import Literal, Optional
import numpy as np

from pypwscf.pot.pseudo import PPDataLocal, PPDataNonLocal
from pypwscf.ele import ElectronDen, ElectronWfcBgrp

from .read_inp import PWscfIn

RANDOMIZE_AMP = 0.05


def initialize_rho(rho: ElectronDen, l_pploc: list[PPDataLocal], pwin: PWscfIn):
    numspin = rho.numspin
    grho = rho.grho
    startingpot = pwin.electrons.startingpot
    if startingpot == "atomic":
        if numspin == 2:
            startingmag = pwin.system.starting_magnetization
            rho_g = np.zeros((2, grho.numg), dtype="c16")
            for isp, pploc in enumerate(l_pploc):
                rho_g[0] += pploc.rhoatomic_g * (1 + startingmag[isp + 1]) / 2
                rho_g[1] += pploc.rhoatomic_g * (1 - startingmag[isp + 1]) / 2
        else:
            rho_g = sum(pploc.rhoatomic_g for pploc in l_pploc).reshape(1, grho.numg)
        rho.update_rho(rho_g)

    elif startingpot == "file":
        raise NotImplementedError("startingpot='file' not yet implemented")

    return rho


def initialize_wfc(
    l_wfc: list[ElectronWfcBgrp],
    l_ppnl: list[PPDataNonLocal],
    pwin: PWscfIn,
    seed: Optional[int] = None,
):
    startingwfc = pwin.electrons.startingwfc
    if startingwfc == "random":
        for wfc in l_wfc:
            wfc.init_random_wfc()
    else:
        raise NotImplementedError("startingwfc='random' not yet implemented")
    return l_wfc

