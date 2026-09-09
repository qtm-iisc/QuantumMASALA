"""Simple, DFT-free unit tests for `qtm.dft.kswfn.KSWfn`: the `gcryst2int`
G-vector hash (injectivity), `overlap`'s reduction to the ordinary plane-wave
inner product for a same-k/zero-umklapp pair, `indices_occupied`/
`indices_empty`'s occupation threshold, and `compute_rho`'s total charge.

Uses a plain synthetic cubic `GkSpace` (as in '../gspace_tests/
test_gspace.py') -- no crystal, no pseudopotential, no SCF.
"""
import numpy as np

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.dft import KSWfn
from qtm.gspace import GkSpace, GSpace
from qtm.lattice import ReciLattice

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gwfn = GSpace(recilat, 10.0)
gkspc = GkSpace(gwfn, (0.13, 0.27, -0.08))


def _make_kswfn(numbnd=4):
    kswfn = KSWfn(gkspc, k_weight=1.0, numbnd=numbnd, is_noncolin=False)
    kswfn.init_random()
    return kswfn


# ----- gcryst2int -------------------------------------------------------------
def test_gcryst2int_is_injective_with_zero_umklapp():
    kswfn = _make_kswfn()
    h = kswfn.gcryst2int(gkspc, [0, 0, 0])
    assert h.shape == (gkspc.size_g,)
    assert len(np.unique(h)) == gkspc.size_g  # no two G-vectors hash to the same value
    assert np.all(h >= 0) and np.all(h < np.prod(gkspc.grid_shape))


def test_gcryst2int_changes_with_umklapp_shift():
    kswfn = _make_kswfn()
    h0 = kswfn.gcryst2int(gkspc, [0, 0, 0])
    h1 = kswfn.gcryst2int(gkspc, [1, 0, 0])
    assert not np.array_equal(h0, h1)
    # still injective after the shift
    assert len(np.unique(h1)) == gkspc.size_g


# ----- overlap -----------------------------------------------------------------
def test_overlap_reduces_to_vdot_for_same_k_zero_umklapp():
    kswfn = _make_kswfn()
    kswfn.evc_gk.normalize()
    bands = list(range(kswfn.numbnd))

    ov = kswfn.overlap(kswfn, bands, bands, [0, 0, 0])
    manual = kswfn.evc_gk.vdot(kswfn.evc_gk)
    assert np.allclose(ov, manual, atol=1e-8)


def test_overlap_subset_of_bands():
    kswfn = _make_kswfn(numbnd=5)
    kswfn.evc_gk.normalize()
    bra_bands, ket_bands = [0, 2], [1, 3, 4]

    ov = kswfn.overlap(kswfn, bra_bands, ket_bands, [0, 0, 0])
    manual = kswfn.evc_gk[bra_bands].vdot(kswfn.evc_gk[ket_bands])
    assert ov.shape == (2, 3)
    assert np.allclose(ov, manual, atol=1e-8)


# ----- indices_occupied / indices_empty -----------------------------------------
def test_indices_occupied_and_empty_use_half_occupation_threshold():
    kswfn = _make_kswfn(numbnd=4)
    kswfn.occ[:] = [1.0, 0.5, 0.4999, 0.0]
    assert kswfn.indices_occupied == (0, 1)
    assert kswfn.indices_empty == (2, 3)


# ----- compute_rho ---------------------------------------------------------------
def test_compute_rho_integrates_to_sum_of_occupations():
    numbnd = 3
    kswfn = _make_kswfn(numbnd=numbnd)
    occ = np.array([1.0, 0.5, 0.0])
    kswfn.occ[:] = occ

    rho = kswfn.compute_rho()
    assert np.isclose(rho.integrate_unitcell(), np.sum(occ), atol=1e-8)
