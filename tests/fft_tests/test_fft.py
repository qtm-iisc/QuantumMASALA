"""Simple, DFT-free unit tests for `qtm.fft`: the `check_g_idxgrid`
bookkeeping validator, and `FFT3DFull.r2g`/`g2r`'s exact normalization
convention, checked directly against `numpy.fft.fftn`/`ifftn`.

Uses the "numpy" backend explicitly to sidestep any optional FFT libraries
(pyFFTW/MKL/CuPy) -- no crystal/GSpace/SCF involved at all.
"""
import numpy as np
import pytest

from qtm import qtmconfig

qtmconfig.set_gpu(False)

from qtm.fft.base import check_g_idxgrid
from qtm.fft.full import FFT3DFull

shape = (3, 4, 5)
size = int(np.prod(shape))
full_idxgrid = np.arange(size, dtype="i8")

rng = np.random.default_rng(0)


# ----- check_g_idxgrid --------------------------------------------------------
def test_check_g_idxgrid_accepts_valid_input():
    check_g_idxgrid(shape, full_idxgrid)  # must not raise
    check_g_idxgrid(shape, np.array([0, 3, size - 1], dtype="i8"))


def test_check_g_idxgrid_rejects_duplicates():
    bad = np.array([0, 1, 1], dtype="i8")
    with pytest.raises(AssertionError):
        check_g_idxgrid(shape, bad)


def test_check_g_idxgrid_rejects_negative_or_out_of_bounds():
    with pytest.raises(AssertionError):
        check_g_idxgrid(shape, np.array([-1, 0, 1], dtype="i8"))
    with pytest.raises(AssertionError):
        check_g_idxgrid(shape, np.array([0, size], dtype="i8"))  # size is out of bounds


# ----- FFT3DFull.r2g / g2r: exact convention against numpy.fft --------------
def _random_grid():
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_r2g_matches_unnormalized_forward_fft():
    fft = FFT3DFull(shape, full_idxgrid, normalise_idft=True, backend="numpy")
    arr_r = _random_grid()
    arr_g = np.empty(size, dtype="c16")
    fft.r2g(arr_r, arr_g)

    expected = np.fft.fftn(arr_r, norm=None).reshape(-1)  # unnormalized forward FFT
    assert np.allclose(arr_g, expected, atol=1e-10)


@pytest.mark.parametrize("normalise_idft,norm", [(True, "backward"), (False, "forward")])
def test_g2r_matches_ifft_with_matching_norm_convention(normalise_idft, norm):
    fft = FFT3DFull(shape, full_idxgrid, normalise_idft=normalise_idft, backend="numpy")
    arr_g = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    arr_r = np.empty(shape, dtype="c16")
    fft.g2r(arr_g, arr_r)

    expected = np.fft.ifftn(arr_g.reshape(shape), norm=norm)
    assert np.allclose(arr_r, expected, atol=1e-10)


def test_round_trip_normalised_idft_is_identity():
    fft = FFT3DFull(shape, full_idxgrid, normalise_idft=True, backend="numpy")
    arr_g = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    arr_r = np.empty(shape, dtype="c16")
    fft.g2r(arr_g, arr_r)
    arr_g_back = np.empty(size, dtype="c16")
    fft.r2g(arr_r, arr_g_back)
    assert np.allclose(arr_g_back, arr_g, atol=1e-10)


def test_round_trip_unnormalised_idft_scales_by_grid_size():
    # normalise_idft=False (what qtm.gspace.gkspc.GkSpace uses for
    # wavefunctions) makes g2r an UNnormalized inverse FFT, so composing it
    # with r2g's unnormalized forward FFT scales by the grid size instead of
    # reproducing the input -- see qtm.containers' 'test_wavefun_round_trip_
    # scales_by_size_r' for the same convention exercised through the
    # container API.
    fft = FFT3DFull(shape, full_idxgrid, normalise_idft=False, backend="numpy")
    arr_g = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    arr_r = np.empty(shape, dtype="c16")
    fft.g2r(arr_g, arr_r)
    arr_g_back = np.empty(size, dtype="c16")
    fft.r2g(arr_r, arr_g_back)
    assert np.allclose(arr_g_back, size * arr_g, atol=1e-8)


def test_r2g_g2r_restricted_to_idxgrid_subset():
    # A truncated 'idxgrid' (as GSpace/GkSpace use for their G-vector shell)
    # must still round-trip correctly: energies/components outside the
    # subset are simply treated as zero.
    idxgrid = np.array([0, 3, 7, size - 1], dtype="i8")
    fft = FFT3DFull(shape, idxgrid, normalise_idft=True, backend="numpy")

    arr_g = rng.standard_normal(len(idxgrid)) + 1j * rng.standard_normal(len(idxgrid))
    arr_r = np.empty(shape, dtype="c16")
    fft.g2r(arr_g, arr_r)

    # Manually zero-pad to the full grid and inverse-FFT for comparison
    full = np.zeros(size, dtype="c16")
    full[idxgrid] = arr_g
    expected_r = np.fft.ifftn(full.reshape(shape), norm="backward")
    assert np.allclose(arr_r, expected_r, atol=1e-10)

    arr_g_back = np.empty(len(idxgrid), dtype="c16")
    fft.r2g(arr_r, arr_g_back)
    assert np.allclose(arr_g_back, arr_g, atol=1e-10)
