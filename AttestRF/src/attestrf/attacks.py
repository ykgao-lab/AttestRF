"""Controlled signal and landmark transformations for stress testing."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def complex_fir(
    signals: torch.Tensor,
    taps: int = 9,
    scale: float = 0.15,
    seed: int = 0,
) -> torch.Tensor:
    """Apply a deterministic complex FIR to [N,2,L] IQ packets.

    This is a stress-test transformation, not a physical RF replay.  The
    coefficients are generated once per call and the output is RMS-normalized
    to keep the perturbation from being a trivial energy cue.
    """

    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError("signals must have shape [N, 2, L]")
    taps = int(taps)
    if taps < 1 or taps % 2 == 0:
        raise ValueError("taps must be a positive odd integer")
    generator = torch.Generator(device=signals.device)
    generator.manual_seed(int(seed))
    coeff = torch.randn(
        taps, device=signals.device, generator=generator, dtype=signals.dtype
    )
    coeff = coeff + 1j * torch.randn(
        taps, device=signals.device, generator=generator, dtype=signals.dtype
    )
    coeff = coeff / torch.linalg.vector_norm(coeff).clamp_min(1e-8) * float(scale)
    value = signals[:, 0] + 1j * signals[:, 1]
    kernel = coeff.reshape(1, 1, -1)
    padded = torch.nn.functional.pad(
        value[:, None], (taps // 2, taps // 2), mode="reflect"
    )
    # Complex convolution is supported by the CUDA builds used here.
    output = torch.nn.functional.conv1d(padded, kernel).squeeze(1)
    output = output + value
    rms = torch.sqrt(
        torch.mean(output.abs().square(), dim=1, keepdim=True).clamp_min(1e-8)
    )
    output = output / rms
    return torch.stack((output.real, output.imag), dim=1)


def landmark_replacement(
    observations: dict[str, torch.Tensor],
    source_id: str,
    targets: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Replace selected authenticated landmark observations with one source."""

    if source_id not in observations:
        raise KeyError(source_id)
    result = {key: value.clone() for key, value in observations.items()}
    for key in targets or list(result):
        if key in result:
            result[key] = result[source_id].clone()
    return result


def landmark_mixture(
    observations: dict[str, torch.Tensor],
    source_id: str,
    fraction: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Mix a fraction of every landmark with a replacement landmark."""

    if source_id not in observations:
        raise KeyError(source_id)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    result = {key: value.clone() for key, value in observations.items()}
    for key, value in result.items():
        count = int(round(len(value) * fraction))
        if count:
            result[key][:count] = observations[source_id][:count]
    return result


def embedding_action_attack(
    model, canonical: torch.Tensor, code: torch.Tensor
) -> torch.Tensor:
    """Apply a learned transport action to canonical embeddings.

    The result is intentionally an embedding-space proxy for an RF acquisition
    action.  It isolates the mechanism and is never reported as a physical
    over-the-air attack.
    """

    if code.ndim == 1:
        code = code.expand(len(canonical), -1)
    return model.transport(canonical, code)
