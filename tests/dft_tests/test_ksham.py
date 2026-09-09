"""Simple, DFT-free unit tests for `qtm.dft.ksham.KSHam`: constructor
validation, the exact `ke_gk` kinetic-energy formula, `vnl_diag`
accumulation across multiple nonlocal projector generators, `h_psi`'s
linearity, and the `vloc.shape == ()` vs `(1,)` equivalence for the
non-noncolinear case.

Uses a plain synthetic cubic `GkSpace` (as in '../gspace_tests/
test_gspace.py') for the validation/`ke_gk`/shape-equivalence tests (no
crystal/pseudopotential needed there), and the real diamond-Si UPF file
(as in '../dft_tests/test_eigsolve_pseudopotential.py') for the
`vnl_diag`/`h_psi` linearity tests, which need genuine nonlocal projectors.
"""
import os

import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.containers import get_FieldR, get_WavefunG
from qtm.crystal import BasisAtoms, Crystal
from qtm.dft import KSHam
from qtm.gspace import GkSpace, GSpace
from qtm.lattice import RealLattice, ReciLattice
from qtm.pseudo import NonlocGenerator, UPFv2Data
from qtm.pseudo.loc import loc_generate_pot_rhocore

# ----- synthetic cubic GkSpace: validation / ke_gk / shape equivalence ------
recilat_cubic = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gwfn = GSpace(recilat_cubic, 10.0)
gkspc = GkSpace(gwfn, (0.13, 0.27, -0.08))
FieldR = get_FieldR(gwfn)


def test_rejects_non_gkspace():
    with pytest.raises(TypeError):
        KSHam("not_a_gkspace", False, FieldR.zeros(()), [])


def test_rejects_non_bool_is_noncolin():
    with pytest.raises(TypeError):
        KSHam(gkspc, "not_a_bool", FieldR.zeros(()), [])


def test_rejects_non_fieldr_vloc():
    with pytest.raises(TypeError):
        KSHam(gkspc, False, np.zeros(10), [])


def test_rejects_vloc_on_mismatched_gspace():
    other_gwfn = GSpace(gwfn.recilat, 20.0)
    other_vloc = get_FieldR(other_gwfn).zeros(())
    with pytest.raises(ValueError):
        KSHam(gkspc, False, other_vloc, [])


def test_rejects_wrong_vloc_shape():
    with pytest.raises(ValueError):
        KSHam(gkspc, False, FieldR.zeros(2), [])  # not noncolin: only () or (1,) valid
    with pytest.raises(ValueError):
        KSHam(gkspc, True, FieldR.zeros(()), [])  # noncolin: () is not valid


def test_rejects_non_nonlocgenerator_l_nloc():
    with pytest.raises(TypeError):
        KSHam(gkspc, False, FieldR.zeros(()), [1, 2, 3])


def test_ke_gk_matches_exact_formula():
    ksham = KSHam(gkspc, is_noncolin=False, vloc=FieldR.zeros(()), l_nloc=[])
    assert np.allclose(ksham.ke_gk.data, (0.5 * gkspc.gk_norm2).astype("c16"))


def test_vloc_shape_unit_and_scalar_are_equivalent():
    v0 = 0.1
    vloc_scalar = FieldR.zeros(())
    vloc_scalar.data[:] = v0
    vloc_unit = FieldR.zeros(1)
    vloc_unit.data[:] = v0

    ksham_scalar = KSHam(gkspc, False, vloc_scalar, [])
    ksham_unit = KSHam(gkspc, False, vloc_unit, [])

    WavefunG = get_WavefunG(gkspc, 1)
    rng = np.random.default_rng(0)
    psi = WavefunG.empty(2)
    psi.data[:] = rng.standard_normal((2, gkspc.size_g)) + 1j * rng.standard_normal(
        (2, gkspc.size_g)
    )

    hpsi_scalar = WavefunG.empty(2)
    ksham_scalar.h_psi(psi, hpsi_scalar)
    hpsi_unit = WavefunG.empty(2)
    ksham_unit.h_psi(psi, hpsi_unit)
    assert np.allclose(hpsi_scalar.data, hpsi_unit.data)


# ----- real Si pseudopotential: vnl_diag accumulation / h_psi linearity ----
reallat = RealLattice.from_alat(
    alat=10.2, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]
)
si_oncv = UPFv2Data.from_file(
    os.path.join(
        os.path.dirname(__file__), "..", "system_tests", "Si_ONCV_PBE-1.2.upf"
    )
)
si_atoms = BasisAtoms(
    "si",
    si_oncv,
    28.086,
    reallat,
    np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]).T,
)
crystal = Crystal(reallat, [si_atoms])
si_gwfn = GSpace(crystal.recilat, 16.0)
si_gkspc = GkSpace(si_gwfn, (0.13, 0.27, -0.08))

v_ion, _rho_core = loc_generate_pot_rhocore(si_atoms, si_gwfn)
si_vloc = v_ion.to_r() / si_gwfn.size_r


def test_vnl_diag_accumulates_linearly_across_generators():
    ksham_single = KSHam(
        si_gkspc, False, si_vloc, [NonlocGenerator(si_atoms, si_gwfn)]
    )
    ksham_double = KSHam(
        si_gkspc,
        False,
        si_vloc,
        [NonlocGenerator(si_atoms, si_gwfn), NonlocGenerator(si_atoms, si_gwfn)],
    )
    assert np.allclose(ksham_double.vnl_diag.data, 2 * ksham_single.vnl_diag.data)


def test_h_psi_is_linear():
    ksham = KSHam(si_gkspc, False, si_vloc, [NonlocGenerator(si_atoms, si_gwfn)])
    WavefunG = get_WavefunG(si_gkspc, 1)
    rng = np.random.default_rng(1)

    def rand_wfn():
        wfn = WavefunG.empty(3)
        wfn.data[:] = rng.standard_normal(
            (3, si_gkspc.size_g)
        ) + 1j * rng.standard_normal((3, si_gkspc.size_g))
        return wfn

    a, b = rand_wfn(), rand_wfn()
    ha, hb, hab = WavefunG.empty(3), WavefunG.empty(3), WavefunG.empty(3)
    ksham.h_psi(a, ha)
    ksham.h_psi(b, hb)
    ksham.h_psi(a + b, hab)
    assert np.allclose(hab.data, ha.data + hb.data, atol=1e-6)
