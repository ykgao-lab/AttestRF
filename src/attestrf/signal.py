"""Small, dependency-free transforms for complex baseband packets."""

from __future__ import annotations

import numpy as np


def global_iq_standardize(value: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Standardize one ``[2, samples]`` I/Q packet with a shared mean and scale."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"expected I/Q input [2, L], found {tuple(array.shape)}")
    scale = max(float(array.std()), float(epsilon))
    return ((array - float(array.mean())) / scale).astype(np.float32, copy=False)


def random_complex_fir(
    value: np.ndarray,
    taps: int = 30,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a length-preserving random complex FIR perturbation."""

    if int(taps) < 1:
        raise ValueError("taps must be positive")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"expected I/Q input [2, L], found {tuple(array.shape)}")
    normal = np.random.normal if rng is None else rng.normal
    mean = 1.0 / (2.0 * int(taps))
    deviation = float(np.sqrt(1.0 / (2.0 * int(taps) + 1.0)))
    coefficients = normal(mean, deviation, size=int(taps)) + 1j * normal(
        mean, deviation, size=int(taps)
    )
    samples = array[0] + 1j * array[1]
    full = np.convolve(samples, coefficients, mode="full")
    start = (len(coefficients) - 1) // 2
    filtered = full[start : start + len(samples)]
    return np.stack((filtered.real, filtered.imag)).astype(np.float32, copy=False)


class IQTransform:
    """Packet standardization with an optional training-only FIR perturbation."""

    def __init__(
        self,
        random_filter_taps: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.random_filter_taps = (
            None if random_filter_taps is None else int(random_filter_taps)
        )
        self.rng = rng

    def reseed(self, seed: int) -> None:
        """Reset the augmentation stream, for example inside a loader worker."""

        self.rng = np.random.default_rng(int(seed))

    def __call__(self, value: np.ndarray) -> np.ndarray:
        output = global_iq_standardize(value)
        if self.random_filter_taps is not None:
            output = random_complex_fir(output, self.random_filter_taps, self.rng)
        return output
