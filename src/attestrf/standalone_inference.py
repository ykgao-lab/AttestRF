"""Encoder, prototype classifier, and invertible feature correction models."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .encoder import IQResNet1D, NormalizedPrototypes


class AdditiveAction(nn.Module):
    """Code-conditioned translation in the encoder feature space."""

    def __init__(self, embedding_dim: int, code_dim: int = 1):
        super().__init__()
        if embedding_dim < 1 or code_dim < 1:
            raise ValueError("embedding and code dimensions must be positive")
        self.embedding_dim = int(embedding_dim)
        self.code_dim = int(code_dim)
        self.bias = nn.Linear(self.code_dim, self.embedding_dim, bias=False)

    def forward(self, canonical: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return canonical + self.bias(code.tanh())

    def inverse(self, observed: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return observed - self.bias(code.tanh())


class LowRankAction(nn.Module):
    """Bounded feature-space action with an analytic inverse.

    With bounded columns P and Q and rho < 1, the linear perturbation has
    operator norm smaller than one for every finite code.  The small system
    used by the inverse therefore remains nonsingular.
    """

    def __init__(self, embedding_dim: int, code_dim: int = 1, rho: float = 0.8):
        super().__init__()
        if not 1 <= code_dim <= embedding_dim:
            raise ValueError(
                "code dimension must lie between 1 and embedding dimension"
            )
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        self.embedding_dim = int(embedding_dim)
        self.code_dim = int(code_dim)
        self.rho = float(rho)
        self.P = nn.Parameter(torch.empty(embedding_dim, code_dim))
        self.Q = nn.Parameter(torch.empty(embedding_dim, code_dim))
        self.B = nn.Parameter(torch.empty(embedding_dim, code_dim))
        nn.init.orthogonal_(self.P)
        nn.init.orthogonal_(self.Q)
        nn.init.normal_(self.B, std=0.02)

    @staticmethod
    def _bounded(matrix: torch.Tensor) -> torch.Tensor:
        return matrix / torch.linalg.matrix_norm(matrix, ord="fro").clamp_min(1.0)

    def _bounded_parameters(self):
        # This public numerical interface is also consumed by the continuous
        # interval optimizer in attestrf.core.
        return self._bounded(self.P), self._bounded(self.Q), self.B

    def forward(self, canonical: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        code = torch.broadcast_to(code, canonical.shape[:-1] + (self.code_dim,))
        p, q, b = self._bounded_parameters()
        t = code.tanh()
        return canonical + ((canonical @ q) * (self.rho * t)) @ p.T + t @ b.T

    def inverse(self, observed: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        code = torch.broadcast_to(code, observed.shape[:-1] + (self.code_dim,))
        p, q, b = self._bounded_parameters()
        t = code.tanh()
        centered = observed - t @ b.T
        diagonal = self.rho * t
        identity = torch.eye(
            self.code_dim, dtype=observed.dtype, device=observed.device
        )
        system = identity + diagonal.unsqueeze(-1) * (q.T @ p)
        right_hand_side = (centered @ q * diagonal).unsqueeze(-1)
        correction = torch.linalg.solve(system, right_hand_side).squeeze(-1) @ p.T
        return centered - correction

    def condition_numbers(self, code: torch.Tensor) -> torch.Tensor:
        p, q, _ = self._bounded_parameters()
        identity = torch.eye(self.embedding_dim, dtype=code.dtype, device=code.device)
        update = torch.einsum("dk,...k,ek->...de", p, self.rho * code.tanh(), q)
        return torch.linalg.cond(identity + update)


class IdentificationModel(nn.Module):
    """Encoder, prototypes, and feature-space nuisance action for inference.

    The receiver table supplies source-receiver codes during training.
    Deployment uses an explicit code fitted from authenticated references.
    """

    def __init__(
        self,
        num_training_identities: int,
        num_training_receivers: int,
        embedding_dim: int = 64,
        receiver_code_dim: int = 1,
        transport: Literal["additive", "lowrank"] = "lowrank",
        input_channels: int = 2,
        temperature: float = 0.1,
        backbone_widths: tuple[int, ...] = (32, 64, 128, 128),
        receiver_code_init_std: float = 0.0,
    ):
        super().__init__()
        if transport not in {"additive", "lowrank"}:
            raise ValueError("transport must be 'additive' or 'lowrank'")
        if input_channels != 2:
            raise ValueError("the I/Q encoder expects two input channels")
        if receiver_code_init_std < 0.0:
            raise ValueError("receiver code initialization scale must be nonnegative")
        self.encoder = IQResNet1D(input_channels, embedding_dim, widths=backbone_widths)
        self.prototypes = NormalizedPrototypes(
            num_training_identities, embedding_dim, temperature
        )
        self.receiver_codes = nn.Embedding(num_training_receivers, receiver_code_dim)
        if receiver_code_init_std:
            nn.init.normal_(
                self.receiver_codes.weight, std=float(receiver_code_init_std)
            )
        else:
            nn.init.zeros_(self.receiver_codes.weight)
        action_type = AdditiveAction if transport == "additive" else LowRankAction
        self.transport = action_type(embedding_dim, receiver_code_dim)
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
