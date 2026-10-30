"""Tests for landmark action envelopes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from attestrf.standalone_inference import AdditiveAction
from attestrf.standalone_landmarks import (
    LandmarkActionFitter,
    action_is_contained,
    calibration_action_envelope,
)


def test_calibration_envelope_contains_calibration_codes() -> None:
    center = np.array([0.0, 0.2])
    codes = {"a": np.array([0.1, 0.1]), "b": np.array([-0.05, 0.35])}
    envelope = calibration_action_envelope(center, codes, minimum_radius=0.01)
    assert action_is_contained(envelope, codes["a"])
    assert action_is_contained(envelope, codes["b"])
    assert not action_is_contained(envelope, np.array([0.5, 0.2]))


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.receiver_code_dim = 1
        self.transport = AdditiveAction(3)
        with torch.no_grad():
            self.transport.bias.weight.copy_(torch.tensor([[0.2], [0.3], [0.1]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_landmark_fitting_uses_the_frozen_action_dtype(dtype) -> None:
    model = FeatureModel().to(dtype=dtype)
    prototypes = {"a": torch.eye(3)[0], "b": torch.eye(3)[1]}
    with torch.no_grad():
        observations = {
            key: model.transport(
                value.to(dtype)[None], torch.tensor([[0.35]], dtype=dtype)
            ).repeat(8, 1)
            for key, value in prototypes.items()
        }
    before = {key: value.clone() for key, value in model.state_dict().items()}
    fitter = LandmarkActionFitter(
        model,
        prototypes,
        min_landmarks=1,
        trim_fraction=0.0,
        steps=180,
        multistart=1,
    )
    fit = fitter.calibrate(observations)
    expected_dtype = np.float32 if dtype == torch.float32 else np.float64
    assert fit.code.dtype == expected_dtype
    assert abs(float(fit.code[0]) - 0.35) < 1e-3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(parameter.grad is None for parameter in model.parameters())
