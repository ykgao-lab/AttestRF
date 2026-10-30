"""Signal-transform regression tests for short and ordinary I/Q inputs."""

from __future__ import annotations

import numpy as np
import pytest

from attestrf.signal import random_complex_fir


@pytest.mark.parametrize("length,taps", [(8, 30), (8, 31), (128, 30), (8, 1)])
def test_random_fir_preserves_input_length_and_dtype(length, taps) -> None:
    signal = np.random.default_rng(4).normal(size=(2, length)).astype(np.float32)
    before = signal.copy()
    filtered = random_complex_fir(signal, taps=taps, rng=np.random.default_rng(5))
    assert filtered.shape == signal.shape
    assert filtered.dtype == np.float32
    assert np.isfinite(filtered).all()
    np.testing.assert_array_equal(signal, before)


def test_random_fir_agrees_with_centered_zero_padded_filtering() -> None:
    signal = np.random.default_rng(2).normal(size=(2, 8)).astype(np.float32)
    padded = np.pad(signal, ((0, 0), (40, 40)))
    short = random_complex_fir(signal, taps=30, rng=np.random.default_rng(3))
    long = random_complex_fir(padded, taps=30, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(short, long[:, 40:48])
