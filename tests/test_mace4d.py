"""
Fast CPU tests for mace4d. No GPU required.

Run from the repo root:  python -m pytest tests/
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mace4d import (
    MACE4DModel,
    construct_time_frames,
    _dejitter_4d_dct,
    _normalize_prior_weights,
)

import mbirjax as mj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_VIEWS = 24
ANGLE_STEP_DEG = 15.0   # 24 views cover 360 degrees


@pytest.fixture(scope="module")
def small_model():
    """Tiny cone-beam model with evenly spaced angles over 360 degrees."""
    angles = np.radians(ANGLE_STEP_DEG) * np.arange(NUM_VIEWS)
    model = mj.ConeBeamModel(
        sinogram_shape=(NUM_VIEWS, 8, 10),
        angles=angles,
        source_detector_dist=100.0,
        source_iso_dist=50.0,
    )
    return model


@pytest.fixture(scope="module")
def small_sino():
    rng = np.random.default_rng(0)
    return rng.uniform(size=(NUM_VIEWS, 8, 10)).astype(np.float32)


# ---------------------------------------------------------------------------
# construct_time_frames
# ---------------------------------------------------------------------------

def test_frame_count_and_views(small_model, small_sino):
    # 120 degree frames advancing 60 degrees: views 0-7, 4-11, 8-15, 12-19, 16-23.
    sino_frames, model_frames = construct_time_frames(
        small_sino, small_model,
        angle_span_per_frame=np.radians(120.0), angle_stride=np.radians(60.0))
    assert len(sino_frames) == len(model_frames) == 5
    assert all(s.shape == (8, 8, 10) for s in sino_frames)
    assert np.array_equal(sino_frames[1], small_sino[4:12])


def test_frame_angles_match_views(small_model, small_sino):
    sino_frames, model_frames = construct_time_frames(
        small_sino, small_model,
        angle_span_per_frame=np.radians(120.0), angle_stride=np.radians(60.0))
    full_angles = small_model.get_all_params()[0]["angles"]
    frame1_angles = model_frames[1].get_all_params()[0]["angles"]
    assert np.allclose(frame1_angles, full_angles[4:12])


def test_span_too_large_raises(small_model, small_sino):
    with pytest.raises(ValueError):
        construct_time_frames(small_sino, small_model,
                              angle_span_per_frame=np.radians(720.0),
                              angle_stride=np.radians(60.0))


# ---------------------------------------------------------------------------
# _dejitter_4d_dct
# ---------------------------------------------------------------------------

def test_dejitter_zeroes_target_bands_only():
    rng = np.random.default_rng(1)
    from scipy.fft import dct
    N, period, bw = 49, 6, 1
    vol = rng.normal(size=(N, 3, 4, 5)).astype(np.float32)
    out = _dejitter_4d_dct(vol, period=period, verbose=False)
    C_in = dct(vol.reshape(N, -1), type=1, norm="ortho", axis=0)
    C_out = dct(out.reshape(N, -1), type=1, norm="ortho", axis=0)
    zeroed = set()
    for h in range(1, period // 2 + 1):
        k0 = int(round(2 * (N - 1) / (period / h)))
        zeroed.update(range(max(0, k0 - bw), min(N, k0 + bw + 1)))
    kept = sorted(set(range(N)) - zeroed)
    assert np.abs(C_out[sorted(zeroed)]).max() < 1e-5
    assert np.allclose(C_out[kept], C_in[kept], atol=1e-4)


def test_dejitter_chunked_equals_whole():
    rng = np.random.default_rng(2)
    vol = rng.normal(size=(30, 4, 5, 6)).astype(np.float32)
    whole = _dejitter_4d_dct(vol, period=6, verbose=False)
    chunked = _dejitter_4d_dct(vol, period=6, chunk_size=2, verbose=False)
    assert np.allclose(whole, chunked, atol=1e-5)


# ---------------------------------------------------------------------------
# _normalize_prior_weights
# ---------------------------------------------------------------------------

def test_prior_weights_scalar_and_list():
    assert _normalize_prior_weights(0.5) == [0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]
    assert _normalize_prior_weights([0.1, 0.2, 0.3]) == [1.0 - 0.6, 0.1, 0.2, 0.3]


@pytest.mark.parametrize("bad", [1.5, -0.1, [0.5, 0.5, 0.5], [0.1, 0.2]])
def test_prior_weights_invalid_raises(bad):
    with pytest.raises(ValueError):
        _normalize_prior_weights(bad)


# ---------------------------------------------------------------------------
# Init-image cache logic
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mace_model(small_model, small_sino):
    sino_frames, model_frames = construct_time_frames(
        small_sino, small_model,
        angle_span_per_frame=np.radians(120.0), angle_stride=np.radians(60.0))
    return MACE4DModel(sino_frames, model_frames, verbose=0)


def test_validate_init_image(mace_model):
    expected = mace_model._expected_init_shape()
    good = np.zeros(expected, dtype=np.float64)
    out = mace_model._validate_init_image(good)
    assert out.dtype == np.float32 and out.shape == expected
    with pytest.raises(ValueError):
        mace_model._validate_init_image(np.zeros((1, 2, 3, 4)))


def test_init_cache_absent_invalid_valid(mace_model, tmp_path):
    d = str(tmp_path)
    # Absent: silent None.
    assert mace_model._load_cached_init(d) is None
    # Present but wrong shape: warns and returns None.
    np.save(os.path.join(d, "init_image.npy"), np.zeros((1, 2, 3, 4), dtype=np.float32))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert mace_model._load_cached_init(d) is None
    assert len(rec) == 1 and "invalid" in str(rec[0].message)
    # Valid: loaded as float32.
    good = np.zeros(mace_model._expected_init_shape(), dtype=np.float32)
    np.save(os.path.join(d, "init_image.npy"), good)
    loaded = mace_model._load_cached_init(d)
    assert loaded is not None and loaded.shape == good.shape
