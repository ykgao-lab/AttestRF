"""Standard source-only classifiers and domain losses used by AttestRF studies."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.autograd import Function

from .encoder import IQResNet1D


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(values: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Preserve the forward values and reverse their gradient by ``strength``."""

    return _GradientReverse.apply(values, float(strength))


class ERMClassifier(nn.Module):
    """IQResNet1D followed by a linear transmitter classifier."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 64,
        input_channels: int = 2,
        widths: Sequence[int] = (32, 64, 128, 128),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = IQResNet1D(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            widths=widths,
            blocks_per_stage=blocks_per_stage,
            dropout=dropout,
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes))

    def forward(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(signals)
        return self.classifier(embedding), embedding


class DANNClassifier(ERMClassifier):
    """ERM classifier with a gradient-reversed source-domain head."""

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        embedding_dim: int = 64,
        input_channels: int = 2,
        widths: Sequence[int] = (32, 64, 128, 128),
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            widths=widths,
            blocks_per_stage=blocks_per_stage,
            dropout=dropout,
        )
        self.receiver_head = nn.Sequential(
            nn.Linear(int(embedding_dim), int(embedding_dim)),
            nn.SiLU(),
            nn.Linear(int(embedding_dim), int(num_domains)),
        )

    def forward(
        self, signals: torch.Tensor, grl_strength: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(signals)
        class_logits = self.classifier(embedding)
        domain_logits = self.receiver_head(gradient_reverse(embedding, grl_strength))
        return class_logits, domain_logits, embedding


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared mean difference between unbiased feature covariance matrices."""

    if len(source) < 2 or len(target) < 2:
        return source.sum() * 0.0
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_covariance = source_centered.T @ source_centered / (len(source) - 1)
    target_covariance = target_centered.T @ target_centered / (len(target) - 1)
    return torch.mean((source_covariance - target_covariance) ** 2)
