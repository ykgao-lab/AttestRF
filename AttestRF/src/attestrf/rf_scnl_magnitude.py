"""Magnitude-aware RF-SCNL representation and receiver-action model.

The model uses the envelope of the two LTF components as its input
representation. No stochastic waveform augmentation is applied.
The action equations are the additive and bounded rank-one actions used by
``attestrf.core`` for interval certification.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn

from .encoder import NormalizedPrototypes, ResidualBlock1D, _group_count
from .standalone_inference import AdditiveAction, LowRankAction


def magnitude_ltf(value: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Convert one I/Q packet to a one-channel, standardized LTF envelope.

    The scalar standardization is computed from both real channels before the
    envelope is formed.  Consequently the transform removes packet gain while
    retaining the relative temporal and quadrature envelope. The output
    preserves the input length. The RF-SCNL dataset adapter supplies the
    first 128 samples of each recording.
    """

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"expected I/Q input [2,L], found {tuple(array.shape)}")
    scale = max(float(array.std()), float(epsilon))
    standardized = (array - float(array.mean())) / scale
    envelope = np.sqrt(
        np.square(standardized[0]) + np.square(standardized[1]) + float(epsilon)
    )
    return envelope[None, :].astype(np.float32, copy=False)


class MagnitudeResNet1D(nn.Module):
    """The compact GroupNorm residual encoder with a one-channel input."""

    def __init__(
        self,
        embedding_dim: int = 64,
        widths: Sequence[int] = (32, 64, 128, 128),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not widths or blocks_per_stage < 1:
            raise ValueError(
                "widths must be nonempty and blocks_per_stage must be positive"
            )
        widths = tuple(int(width) for width in widths)
        self.embedding_dim = int(embedding_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(1, widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(widths[0]), widths[0]),
            nn.SiLU(inplace=True),
        )
        stages: list[nn.Module] = []
        in_channels = widths[0]
        for stage_index, width in enumerate(widths):
            for block_index in range(blocks_per_stage):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(ResidualBlock1D(in_channels, width, stride))
                in_channels = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        hidden = max(128, self.embedding_dim * 2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.embedding_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 1:
            raise ValueError(
                f"MagnitudeResNet1D expects [B,1,L], found {tuple(values.shape)}"
            )
        return self.head(self.pool(self.stages(self.stem(values))))


class MagnitudeIdentificationModel(nn.Module):
    """Magnitude encoder, normalized identity prototypes, and scalar action."""

    def __init__(
        self,
        num_identities: int = 8,
        num_source_domains: int = 2,
        embedding_dim: int = 64,
        receiver_code_dim: int = 1,
        transport: Literal["additive", "lowrank"] = "additive",
        temperature: float = 0.1,
        backbone_widths: Sequence[int] = (32, 64, 128, 128),
        receiver_code_init_std: float = 0.5,
        lowrank_rho: float = 0.8,
    ):
        super().__init__()
        if transport not in {"additive", "lowrank"}:
            raise ValueError("transport must be additive or lowrank")
        if receiver_code_dim != 1:
            raise ValueError("exact interval evaluation requires a scalar action code")
        self.encoder = MagnitudeResNet1D(
            embedding_dim=embedding_dim, widths=backbone_widths
        )
        self.prototypes = NormalizedPrototypes(
            num_identities, embedding_dim, temperature
        )
        self.receiver_codes = nn.Embedding(num_source_domains, receiver_code_dim)
        nn.init.normal_(self.receiver_codes.weight, std=float(receiver_code_init_std))
        if transport == "additive":
            self.transport = AdditiveAction(embedding_dim, receiver_code_dim)
        else:
            self.transport = LowRankAction(
                embedding_dim, receiver_code_dim, rho=lowrank_rho
            )
        self.embedding_dim = int(embedding_dim)
        self.receiver_code_dim = int(receiver_code_dim)

    def encode(self, signals: torch.Tensor) -> torch.Tensor:
        return self.encoder(signals)

    def canonicalize(self, observed: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return self.transport.inverse(observed, code)

    def forward(
        self, signals: torch.Tensor, receiver_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        observed = self.encode(signals)
        code = self.receiver_codes(receiver_ids)
        canonical = self.canonicalize(observed, code)
        return {
            "observed": observed,
            "code": code,
            "canonical": canonical,
            "logits": self.prototypes.logits(canonical),
        }
