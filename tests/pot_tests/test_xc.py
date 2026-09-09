"""Simple, DFT-free unit tests for `qtm.pot.xc`: the pure bookkeeping in
`get_libxc_func`/`check_libxc_func`, and `compute`'s LDA exchange output
against the textbook closed form for a uniform electron gas, checked on a
spatially uniform synthetic density (no gradient/GGA complications).

Uses a plain synthetic cubic `GSpace` (as in '../gspace_tests/test_gspace.py')
-- no crystal, no pseudopotential, no SCF. `get_libxc_func` only needs
lightweight stand-ins with a '.ppdata.libxc_func' attribute, not a real
`Crystal`/`BasisAtoms`.
"""
import types

import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.containers import get_FieldG
from qtm.gspace.gspc import GSpace
from qtm.lattice import ReciLattice
from qtm.pot import xc

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gsp = GSpace(recilat, 10.0)
FieldG = get_FieldG(gsp)


def _species(libxc_func, label="X"):
    ppdata = types.SimpleNamespace(libxc_func=libxc_func, filename=f"{label}.upf")
    return types.SimpleNamespace(label=label, ppdata=ppdata)


# ----- get_libxc_func / check_libxc_func: pure bookkeeping ------------------
def test_get_libxc_func_returns_common_functional():
    crystal = types.SimpleNamespace(
        l_atoms=[
            _species(("gga_x_pbe", "gga_c_pbe"), "A"),
            _species(("gga_x_pbe", "gga_c_pbe"), "B"),
        ]
    )
    assert xc.get_libxc_func(crystal) == ("gga_x_pbe", "gga_c_pbe")


def test_get_libxc_func_raises_on_disagreement():
    crystal = types.SimpleNamespace(
        l_atoms=[
            _species(("gga_x_pbe", "gga_c_pbe"), "A"),
            _species(("lda_x", "lda_c_pz"), "B"),
        ]
    )
    with pytest.raises(ValueError):
        xc.get_libxc_func(crystal)


def test_get_libxc_func_returns_none_when_all_unset():
    crystal = types.SimpleNamespace(l_atoms=[_species(None, "A"), _species(None, "B")])
    assert xc.get_libxc_func(crystal) is None


def test_check_libxc_func_accepts_known_names_and_rejects_unknown():
    xc.check_libxc_func(("lda_x", "lda_c_pz"))  # must not raise
    with pytest.raises(ValueError):
        xc.check_libxc_func(("not_a_real_functional_name",))


# ----- compute(): LDA exchange against the closed-form uniform-gas formula --
def test_lda_exchange_matches_closed_form_on_a_uniform_density():
    # Dirac/Slater exchange: eps_x(rho) = -(3/4)*(3*rho/pi)^(1/3),
    # v_x(rho) = (4/3)*eps_x(rho) (Euler's theorem, since eps_x ~ rho^(1/3)).
    # Exact pointwise for a spatially UNIFORM density -- no gradient terms
    # involved. 'exch_name' and 'corr_name' are passed the same functional
    # name so the result is exactly twice the pure-exchange value, without
    # needing a separate "null" correlation functional.
    rho0 = 0.05

    # qtm.containers' FieldG<->FieldR convention is the literal discrete-FFT
    # one (GSpace uses normalise_idft=True): a uniform real-space value of
    # rho0 corresponds to a G=0 ('data[..., 0]') coefficient of
    # rho0 * gspc.size_r, not rho0 itself.
    rho = FieldG.zeros(1)
    rho.data[0, 0] = rho0 * gsp.size_r
    rhocore = FieldG.zeros(1)

    v_xc, en_xc = xc.compute(rho, rhocore, "lda_x", "lda_x")

    eps_x = -0.75 * (3 * rho0 / np.pi) ** (1 / 3)
    v_x = (4 / 3) * eps_x

    assert np.allclose(v_xc.data, 2 * v_x, atol=1e-10)
    # The real-space unit-cell volume is 'gspc.reallat_cellvol' (dual to
    # 'recilat') -- NOT 'gspc.recilat.cellvol', which is the reciprocal
    # cell's own volume and would be off by a large, grid-dependent factor.
    assert np.isclose(en_xc, 2 * rho0 * eps_x * gsp.reallat_cellvol)
