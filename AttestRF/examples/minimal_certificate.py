"""Run a small certificate calculation on synthetic embeddings."""

from __future__ import annotations

import numpy as np
import torch

from attestrf.core import (
    action_set_from_radius,
    continuous_interval_margin,
    worst_case_margin,
)
from attestrf.standalone_inference import AdditiveAction, LowRankAction


class FeatureModel:
    """Minimal model adapter required by ``worst_case_margin``."""

    def __init__(self, dimension: int = 16):
        self.transport = AdditiveAction(dimension, code_dim=1)
        self.receiver_code_dim = 1

    def canonicalize(self, observed: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return self.transport.inverse(observed, code)


class RankOneFeatureModel(FeatureModel):
    """Minimal adapter using the bounded rank-one receiver action."""

    def __init__(self, dimension: int = 16):
        self.transport = LowRankAction(dimension, code_dim=1)
        self.receiver_code_dim = 1


def main() -> None:
    torch.manual_seed(7)
    model = FeatureModel()
    observed = torch.randn(24, 16)
    prototypes = torch.randn(4, 16)
    action_set = action_set_from_radius(np.array([0.0]), 0.1, grid_points=9)
    result = continuous_interval_margin(
        model, observed, prototypes, center=0.0, radius=0.1
    )
    admitted = result["robust_margin"] > 0.0
    print(f"positive_interval_margin={int(admitted.sum())}/{len(admitted)}")
    print(f"minimum_margin={float(result['robust_margin'].min()):.4f}")

    rank_model = RankOneFeatureModel()
    exact = continuous_interval_margin(
        rank_model, observed, prototypes, center=0.0, radius=0.1
    )
    dense = worst_case_margin(
        rank_model,
        observed,
        prototypes,
        action_set_from_radius(np.array([0.0]), 0.1, grid_points=1001),
    )
    discrepancy = float((exact["robust_margin"] - dense["robust_margin"]).abs().max())
    print(f"rank_one_max_grid_difference={discrepancy:.6f}")
    print(
        "This example checks the interval only; run reference_identification.py for the full rule."
    )


if __name__ == "__main__":
    main()
