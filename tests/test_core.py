"""Tests for the certificate primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from attestrf.core import (
    _quadratic_over_linear_minimum,
    action_set_from_radius,
    continuous_interval_margin,
    worst_case_margin,
)
from attestrf.standalone_inference import AdditiveAction, LowRankAction


class FeatureModel:
    def __init__(self, dimension: int = 4):
        self.transport = AdditiveAction(dimension, code_dim=1)
        self.receiver_code_dim = 1

    def canonicalize(self, observed: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return self.transport.inverse(observed, code)


class RankOneModel(FeatureModel):
    def __init__(self, dimension: int = 4):
        self.transport = LowRankAction(dimension, code_dim=1)
        self.receiver_code_dim = 1


def test_action_set_contains_center_and_endpoints() -> None:
    result = action_set_from_radius(np.array([0.0]), 0.2, grid_points=4)
    values = result.actions[:, 0]
    assert np.isclose(values.min(), -0.2)
    assert np.isclose(values.max(), 0.2)
    assert np.any(np.isclose(values, 0.0))


def test_continuous_margin_returns_finite_values() -> None:
    torch.manual_seed(2)
    model = FeatureModel()
    observed = torch.randn(3, 4)
    prototypes = torch.randn(3, 4)
    output = continuous_interval_margin(model, observed, prototypes, -0.1, 0.1)
    assert output["nominal_pred"].shape == (3,)
    assert torch.isfinite(output["robust_margin"]).all()


@pytest.mark.parametrize("embedding_dtype", [torch.float32, torch.float64])
def test_sampled_margin_accepts_mixed_embedding_and_prototype_dtypes(
    embedding_dtype,
) -> None:
    torch.manual_seed(8)
    model = FeatureModel(3)
    model.transport.to(dtype=embedding_dtype)
    observed = torch.randn(4, 3, dtype=embedding_dtype)
    prototypes = torch.eye(
        3, dtype=(torch.float64 if embedding_dtype == torch.float32 else torch.float32)
    )
    actions = action_set_from_radius([0.0], 0.1, 5)
    mixed = worst_case_margin(model, observed, prototypes, actions)
    matched = worst_case_margin(
        model, observed, prototypes.to(embedding_dtype), actions
    )
    assert mixed["robust_margin"].dtype == embedding_dtype
    torch.testing.assert_close(mixed["robust_margin"], matched["robust_margin"])
    torch.testing.assert_close(mixed["nominal_pred"], matched["nominal_pred"])


def test_rank_one_stationary_evaluation_agrees_with_a_dense_grid() -> None:
    torch.manual_seed(11)
    model = RankOneModel()
    model.transport.double()
    observed = torch.randn(3, 4, dtype=torch.float64)
    prototypes = torch.randn(3, 4, dtype=torch.float64)
    exact = continuous_interval_margin(
        model, observed, prototypes, center=0.2, radius=0.7
    )
    sampled = worst_case_margin(
        model,
        observed,
        prototypes,
        action_set_from_radius([0.2], [0.7], grid_points=1001),
    )
    assert bool((exact["robust_margin"] <= sampled["robust_margin"] + 1e-9).all())
    assert float((exact["robust_margin"] - sampled["robust_margin"]).abs().max()) < 2e-4


def test_additive_margin_matches_direct_endpoint_evaluation_with_intercept() -> None:
    model = FeatureModel(2)
    model.transport.bias = torch.nn.Linear(1, 2, bias=True).double()
    with torch.no_grad():
        model.transport.bias.weight.copy_(torch.tensor([[0.3], [-0.4]]))
        model.transport.bias.bias.copy_(torch.tensor([0.2, -0.1]))
    observed = torch.tensor([[1.0, 0.2], [0.1, 0.9]], dtype=torch.float64)
    prototypes = torch.eye(2, dtype=torch.float64)
    exact = continuous_interval_margin(model, observed, prototypes, 0.2, 0.6)
    sampled = worst_case_margin(
        model, observed, prototypes, action_set_from_radius([0.2], 0.6, 2)
    )
    torch.testing.assert_close(exact["robust_margin"], sampled["robust_margin"])
    torch.testing.assert_close(exact["nominal_scores"], sampled["nominal_scores"])


def test_rank_one_interior_minimum_detects_a_change_missed_by_endpoints() -> None:
    model = RankOneModel(2)
    model.transport.double()
    with torch.no_grad():
        model.transport.P.copy_(torch.tensor([[1.0], [0.0]]))
        model.transport.Q.copy_(torch.tensor([[0.0], [1.0]]))
        model.transport.B.copy_(torch.tensor([[0.0], [1.0]]))
    observed = torch.tensor([[0.04, 0.6]], dtype=torch.float64)
    prototypes = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    radius = float(np.arctanh(0.8))
    result = continuous_interval_margin(model, observed, prototypes, 0.0, radius)
    # The pairwise gap is 0.08 - 0.96*t + 1.6*t^2. Its minimum
    # occurs at t=0.3 and equals -0.064; both endpoints are positive.
    torch.testing.assert_close(
        result["robust_margin"], torch.tensor([-0.064], dtype=torch.float64)
    )
    assert float(result["nominal_margin"][0]) > 0.0
    assert min(0.08 - 0.96 * t + 1.6 * t * t for t in (-0.8, 0.8)) > 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_interval_rejects_nonfinite_features(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        continuous_interval_margin(
            FeatureModel(2), torch.tensor([[value, 1.0]]), torch.eye(2), 0.0, 0.1
        )


def test_zero_width_interval_equals_the_nominal_score_gap() -> None:
    model = RankOneModel(2)
    observed = torch.tensor([[1.0, 0.3], [0.2, 1.0]])
    result = continuous_interval_margin(model, observed, torch.eye(2), 0.1, 0.0)
    torch.testing.assert_close(result["robust_margin"], result["nominal_margin"])


def test_stationary_minimum_retains_small_nonzero_coefficients() -> None:
    def tensor(value):
        return torch.tensor([[value]], dtype=torch.float64)

    result = _quadratic_over_linear_minimum(
        tensor(1e-20),
        tensor(-0.6e-20),
        tensor(0.1e-20),
        tensor(0.0),
        tensor(-0.8),
        tensor(0.8),
    )
    torch.testing.assert_close(result, tensor(0.01e-20), atol=1e-35, rtol=1e-12)
