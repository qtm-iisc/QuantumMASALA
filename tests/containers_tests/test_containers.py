"""Simple, DFT-free unit tests for `qtm.containers`: `BufferType`'s
array-creation/indexing/ufunc-dispatch bookkeeping, the `FieldG`/`FieldR` and
`WavefunG`/`WavefunR` `to_r()`/`to_g()` Fourier-transform wiring, and derived
quantities (`norm2`/`vdot`/`normalize`/`get_density`/`integrate_unitcell`).

Everything here uses a plain synthetic cubic `GSpace`/`GkSpace` (as in
'../gspace_tests/test_gspace.py'), no crystal/pseudopotential/SCF involved.
"""
import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.containers import get_FieldG, get_FieldR, get_WavefunG, get_WavefunR
from qtm.gspace.gkspc import GkSpace
from qtm.gspace.gspc import GSpace
from qtm.lattice import ReciLattice

recilat = ReciLattice.from_tpiba(1.0, (1, 0, 0), (0, 1, 0), (0, 0, 1))
gsp = GSpace(recilat, 10.0)
gkspc = GkSpace(gsp, (0.1, -0.2, 0.05))

FieldG = get_FieldG(gsp)
FieldR = get_FieldR(gsp)
WavefunG = get_WavefunG(gkspc, 1)
WavefunR = get_WavefunR(gkspc, 1)

rng = np.random.default_rng(0)


def _rand_field_g(shape):
    return rng.standard_normal((*shape, gsp.size_g)) + 1j * rng.standard_normal(
        (*shape, gsp.size_g)
    )


def _rand_wavefun_g(shape):
    return rng.standard_normal((*shape, gkspc.size_g)) + 1j * rng.standard_normal(
        (*shape, gkspc.size_g)
    )


# ----- array creation / bookkeeping ------------------------------------------
def test_empty_zeros_shape_and_dtype():
    buf = FieldG.empty((2, 3), dtype="c16")
    assert buf.data.shape == (2, 3, gsp.size_g)
    assert buf.data.dtype == np.dtype("c16")
    assert buf.shape == (2, 3)
    assert buf.rank == 2

    z = FieldG.zeros(4)
    assert np.all(z.data == 0)


def test_from_array_copies_and_validates_basis_size():
    data = _rand_field_g((3,))
    buf = FieldG.from_array(data)
    assert np.array_equal(buf.data, data)
    data[0, 0] = 12345.0  # mutate the original array
    assert buf.data[0, 0] != 12345.0  # 'from_array' must have copied it

    with pytest.raises(AssertionError):
        FieldG.from_array(_rand_field_g((3,))[:, :-1])  # wrong basis size


def test_copy_is_independent():
    buf = FieldG.zeros(2)
    buf2 = buf.copy()
    buf2.data[:] = 1.0
    assert np.all(buf.data == 0)


def test_reshape_preserves_basis_size():
    buf = FieldG.empty(6)
    buf.data[:] = _rand_field_g((6,))
    reshaped = buf.reshape((2, 3))
    assert reshaped.shape == (2, 3)
    assert np.array_equal(reshaped.data, buf.data.reshape(2, 3, gsp.size_g))

    with pytest.raises(Exception):
        buf.reshape((4,))  # 4*basis_size != 6*basis_size elements


def test_getitem_setitem_and_iteration():
    buf = FieldG.empty(3)
    buf.data[:] = _rand_field_g((3,))

    single = buf[1]
    assert single.shape == ()
    assert np.array_equal(single.data, buf.data[1])
    assert buf[0:2].shape == (2,)

    buf[0] = FieldG.zeros(())  # __setitem__ accepts a Buffer instance
    assert np.all(buf.data[0] == 0)

    buf[1] = np.ones(gsp.size_g)  # __setitem__ also accepts a bare array
    assert np.all(buf.data[1] == 1)

    assert len(buf) == 3
    assert [x.shape for x in buf] == [(), (), ()]


# ----- arithmetic / ufunc dispatch -------------------------------------------
def test_arithmetic_operators_match_manual_data_ops():
    a = FieldG.from_array(_rand_field_g((3,)))
    b = FieldG.from_array(_rand_field_g((3,)))

    s = a + b
    assert type(s) is FieldG
    assert np.allclose(s.data, a.data + b.data)
    assert np.allclose((a - b).data, a.data - b.data)
    assert np.allclose((a * 2.0).data, a.data * 2.0)
    assert np.allclose(a.conj().data, a.data.conj())


def test_inplace_add_uses_out_and_stays_buffer_type():
    a = FieldG.empty(3)
    a.data[:] = 1.0
    b = FieldG.empty(3)
    b.data[:] = 2.0
    a += b
    assert type(a) is FieldG
    assert np.allclose(a.data, 3.0)


def test_reduce_over_last_axis_returns_bare_array_otherwise_buffer():
    # Per BufferType's docstring: a reduce that does NOT touch the basis
    # (last) axis stays a Buffer; one that does collapses to a bare array.
    buf = FieldG.from_array(_rand_field_g((3,)))

    reduced_leading = np.add.reduce(buf, axis=0)
    assert type(reduced_leading) is FieldG
    assert reduced_leading.shape == ()

    reduced_basis = np.add.reduce(buf, axis=-1)
    assert type(reduced_basis) is np.ndarray
    assert reduced_basis.shape == (3,)


# ----- to_r()/to_g() Fourier-transform wiring --------------------------------
def test_fieldg_fieldr_round_trip():
    # GSpace uses the standard (normalised-inverse) FFT convention, so this
    # round trip through the container's to_r()/to_g() must be exact.
    fg = FieldG.from_array(_rand_field_g((2,)))
    assert np.allclose(fg.to_r().to_g().data, fg.data, atol=1e-9)


def test_wavefun_round_trip_scales_by_size_r():
    # GkSpace uses the UNnormalised-inverse FFT convention (see
    # qtm.gspace.gkspc.GkSpace._normalise_idft = False and
    # qtm.fft.full.FFT3DFull.g2r/r2g), so to_r()/to_g() is not a plain round
    # trip like FieldG/FieldR's -- it scales the data by the real-space grid
    # size instead of reproducing it exactly.
    wfn = WavefunG.from_array(_rand_wavefun_g((2,)))
    back = wfn.to_r().to_g()
    assert np.allclose(back.data, gkspc.size_r * wfn.data, atol=1e-6)


# ----- WavefunG derived quantities -------------------------------------------
def test_wavefun_norm2_and_vdot_match_manual_computation():
    wfn = WavefunG.from_array(_rand_wavefun_g((3,)))
    manual_norm2 = np.array([np.vdot(row, row).real for row in wfn.data])
    assert np.allclose(wfn.norm2(), manual_norm2)
    assert np.allclose(wfn.norm(), np.sqrt(manual_norm2))

    other = WavefunG.from_array(_rand_wavefun_g((3,)))
    braket = wfn.vdot(other)
    manual = np.array([[np.vdot(a, b) for b in other.data] for a in wfn.data])
    assert np.allclose(braket, manual, atol=1e-8)


def test_wavefun_normalize_sets_unit_norm():
    wfn = WavefunG.from_array(_rand_wavefun_g((2,)))
    wfn.normalize()
    assert np.allclose(wfn.norm2(), 1.0)


def test_wavefunr_get_density_normalizes_to_one():
    wfn_r = WavefunG.from_array(_rand_wavefun_g((1,))).to_r()

    den = wfn_r.get_density(normalize=True)
    assert np.allclose(den.integrate_unitcell(), 1.0, atol=1e-8)

    den_unnorm = wfn_r.get_density(normalize=False)
    manual = (wfn_r.data.conj() * wfn_r.data).real.reshape(1, 1, gkspc.size_r)
    assert np.allclose(den_unnorm.data, manual)


# ----- FieldR.integrate_unitcell --------------------------------------------
def test_integrate_unitcell_without_other_reduces_only_basis_axis():
    f = FieldR.from_array(rng.standard_normal((2, gsp.size_r)) + 0j)
    expected = np.sum(f.data, axis=-1) * gsp.reallat_dv
    assert np.allclose(f.integrate_unitcell(), expected)


def test_integrate_unitcell_with_other_reduces_leading_axis_too():
    # Unlike 'other=None' (which reduces only the basis axis, giving one
    # value per leading index), passing 'other' first reduces the basis
    # axis and then applies a SECOND np.sum with 'axis' (default -1) --
    # which, since the basis axis is already gone, ends up collapsing the
    # leading axis as well. Its one caller (qtm.tddft_gamma.optical.
    # compute_dipole) relies on this by always passing an explicit 'axis='
    # to target a specific surviving axis; pinned down here so this
    # easy-to-trip-over asymmetry doesn't change silently.
    f = FieldR.from_array(rng.standard_normal((2, gsp.size_r)) + 0j)
    other = rng.standard_normal((2, gsp.size_r))

    per_leading_index = np.sum(f.data * other, axis=-1) * gsp.reallat_dv  # shape (2,)
    result = f.integrate_unitcell(other=other)  # default axis=-1
    assert np.shape(result) == ()
    assert np.isclose(result, np.sum(per_leading_index))
