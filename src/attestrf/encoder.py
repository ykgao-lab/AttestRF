"""Residual I/Q encoder and normalized device prototypes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, maximum: int = 16) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = self.activation(self.norm1(self.conv1(values)))
        values = self.norm2(self.conv2(values))
        return self.activation(values + residual)


class IQResNet1D(nn.Module):
    """Group-normalized residual encoder for two-channel I/Q signals."""

    def __init__(
        self,
        input_channels: int = 2,
        embedding_dim: int = 32,
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
            nn.Conv1d(
                input_channels,
                widths[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
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
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(f"IQResNet1D expects [B,2,L], found {tuple(values.shape)}")
        return self.head(self.pool(self.stages(self.stem(values))))


class NormalizedPrototypes(nn.Module):
    def __init__(self, count: int, embedding_dim: int, temperature: float = 0.1):
        super().__init__()
        if count < 2 or temperature <= 0:
            raise ValueError(
                "prototype count must exceed one and temperature must be positive"
            )
        self.weight = nn.Parameter(torch.randn(int(count), int(embedding_dim)) * 0.05)
        self.temperature = float(temperature)

    def normalized(self) -> torch.Tensor:
        return F.normalize(self.weight, dim=-1)

    def logits(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(embedding, dim=-1) @ self.normalized().T / self.temperature


def row_local_standardize(signals: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError(
            f"signals must have shape [B,2,L], found {tuple(signals.shape)}"
        )
    flat = signals.flatten(start_dim=1)
    mean = flat.mean(dim=1, keepdim=True)
    standard_deviation = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(
        epsilon
    )
    return ((flat - mean) / standard_deviation).reshape_as(signals)


def supervised_contrastive_loss(
    embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    if len(embedding) <= 1:
        return embedding.sum() * 0.0
    values = F.normalize(embedding, dim=-1)
    logits = values @ values.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    diagonal = torch.eye(len(values), dtype=torch.bool, device=values.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    denominator_mask = ~diagonal
    exponential = torch.exp(logits) * denominator_mask
    log_probability = logits - torch.log(
        exponential.sum(dim=1, keepdim=True).clamp_min(1e-12)
    )
    valid = positive.sum(dim=1) > 0
    if not valid.any():
        return embedding.sum() * 0.0
    mean_log_probability = (positive * log_probability).sum(dim=1) / positive.sum(
        dim=1
    ).clamp_min(1)
    return -mean_log_probability[valid].mean()
