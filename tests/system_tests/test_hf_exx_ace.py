import numpy as np

from qtm.constants import RYDBERG
from qtm.dft import KSWfn
from qtm.dft.exchange import Exchange
from qtm.gspace import GSpace, GkSpace
from qtm.lattice import RealLattice, ReciLattice


def _make_kswfn():
    reallat = RealLattice.from_alat(
        8.0,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    recilat = ReciLattice.from_reallat(reallat)

    gwfn = GSpace(
        recilat,
        8.0 * RYDBERG,
    )

    gkspc = GkSpace(
        gwfn,
        (0.0, 0.0, 0.0),
    )

    kswfn = KSWfn(
        gkspc,
        k_weight=1.0,
        numbnd=2,
        is_noncolin=False,
    )

    ncoeff = kswfn.evc_gk.data.reshape(2, -1).shape[1]
    x = np.arange(1, ncoeff + 1, dtype=float)

    coeff = np.column_stack(
        [
            np.sin(0.37 * x) + 1j * np.cos(0.23 * x),
            np.cos(0.41 * x) + 1j * np.sin(0.31 * x),
        ]
    )

    q, _ = np.linalg.qr(coeff)

    kswfn.evc_gk.data[...] = q.T.reshape(
        kswfn.evc_gk.data.shape
    )
    kswfn.occ[:] = 2.0

    return reallat, gkspc, kswfn


def test_ace_matches_full_exchange_on_occupied_wavefunctions():
    reallat, gkspc, kswfn = _make_kswfn()

    full = Exchange(
        gkspc=gkspc,
        psi=kswfn.evc_gk,
        occ=kswfn.occ,
        volume=reallat.cellvol,
        method="none",
        use_ace=False,
    )

    ace = Exchange(
        gkspc=gkspc,
        psi=kswfn.evc_gk,
        occ=kswfn.occ,
        volume=reallat.cellvol,
        method="none",
        use_ace=True,
    )

    for ibnd in range(kswfn.numbnd):
        psi = kswfn.evc_gk[ibnd]

        vx_full = full.apply_fock(psi)
        vx_ace = ace.apply_fock(psi)

        np.testing.assert_allclose(
            vx_ace.data,
            vx_full.data,
            rtol=1e-10,
            atol=1e-12,
        )
