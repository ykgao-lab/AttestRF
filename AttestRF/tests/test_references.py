"""Tests of independent-reference criteria and final identification."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from attestrf.references import (
    ReferenceStratum,
    ReferenceThresholds,
    evaluate_reference_gate,
    fit_reference_stratum,
    identify_with_reference_check,
    known_reference_accuracy,
)
from attestrf.standalone_inference import AdditiveAction
from attestrf.standalone_landmarks import LandmarkFit, calibration_action_envelope


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.receiver_code_dim = 1
        self.transport = AdditiveAction(3)
        with torch.no_grad():
            self.transport.bias.weight.copy_(torch.tensor([[0.2], [0.3], [0.1]]))

    def canonicalize(self, observed, code):
        return self.transport.inverse(observed, code)


def make_stratum(name="cell", contained=(True, True)) -> ReferenceStratum:
    fit = LandmarkFit(np.array([0.0]), ["c"], {"c": 0.01}, ["c"], 0.01, 1.0, 0.0, 1)
    envelope = calibration_action_envelope(np.array([0.0]), {"k": np.array([0.2])})
    ids = ("v0", "v1")
    return ReferenceStratum(
        name,
        ("c",),
        ("k",),
        ids,
        envelope,
        fit,
        {},
        dict(zip(ids, contained)),
        {"v0": 0.9, "v1": 0.95},
        {"c": 0.01, "v0": 0.02},
    )


def test_overall_and_stratum_containment_are_both_required() -> None:
    a = make_stratum("a", (True, False))
    b = make_stratum("b")
    permissive = ReferenceThresholds(0.75, {"a": 0.5, "b": 1.0})
    assert evaluate_reference_gate([a, b], permissive).passed
    stricter = replace(permissive, stratum_containment={"a": 1.0})
    assert not evaluate_reference_gate([a, b], stricter).passed
    assert not evaluate_reference_gate(
        [a, b], replace(permissive, overall_containment=1.0)
    ).passed


def test_residual_and_accuracy_components_are_independent() -> None:
    item = make_stratum()
    limits = ReferenceThresholds(
        maximum_residual={"cell": 0.03}, minimum_accuracy={"cell": 0.8}
    )
    assert evaluate_reference_gate([item], limits).passed
    assert not evaluate_reference_gate(
        [item], replace(limits, maximum_residual={"cell": 0.015})
    ).passed
    assert not evaluate_reference_gate(
        [item], replace(limits, minimum_accuracy={"cell": 0.92})
    ).passed


def test_roles_cannot_switch_between_strata() -> None:
    item = make_stratum("a")
    switched = replace(make_stratum("b"), center_ids=("k",), range_ids=("c",))
    with pytest.raises(ValueError, match="retain disjoint roles"):
        evaluate_reference_gate([item, switched], ReferenceThresholds())


def test_failed_reference_gate_withholds_stable_candidate_decisions() -> None:
    model = FeatureModel()
    item = make_stratum(contained=(False, False))
    gate = evaluate_reference_gate([item], ReferenceThresholds())
    result = identify_with_reference_check(
        model, torch.eye(3), torch.eye(3), item, gate
    )
    assert bool((result["robust_margin"] > 0).all())
    assert not bool(result["admitted"].any())
    assert bool((result["identity"] == -1).all())


def test_known_accuracy_competes_with_test_device_source_prototypes() -> None:
    model = FeatureModel()
    # The observation is closer to a test-device prototype than to either
    # reference prototype, so a reference-only comparison would be wrong.
    prototypes = {
        "reference": torch.tensor([1.0, 0.0, 0.0]),
        "other_reference": torch.tensor([-1.0, 0.0, 0.0]),
        "test_device": torch.tensor([1.0, 1.0, 0.0]),
    }
    observed = torch.tensor([[1.0, 1.0, 0.0]])
    assert (
        known_reference_accuracy(
            model, observed, "reference", prototypes, np.array([0.0])
        )
        == 0.0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_fit_does_not_change_weights_or_use_check_to_expand_interval(
    dtype,
) -> None:
    torch.manual_seed(3)
    model = FeatureModel().to(dtype=dtype)
    prototypes = {"c": torch.eye(3)[0], "k": torch.eye(3)[1], "v": torch.eye(3)[2]}
    with torch.no_grad():
        observations = {
            key: model.transport(
                value.to(dtype)[None], torch.tensor([[code]], dtype=dtype)
            )
            .repeat(8, 1)
            .float()
            for (key, value), code in zip(prototypes.items(), (0.1, 0.2, 0.8))
        }
    before = {key: value.clone() for key, value in model.state_dict().items()}
    item = fit_reference_stratum(
        model,
        prototypes,
        observations,
        name="cell",
        center_ids=["c"],
        range_ids=["k"],
        check_ids=["v"],
        steps=180,
        multistart=1,
    )
    assert not item.contained["v"]
    assert float(item.envelope.radius[0]) < 0.2
    assert float(item.individual_fits["v"].code[0]) > 0.6
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fitting_rejects_overlapping_roles() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        fit_reference_stratum(
            FeatureModel(),
            {},
            {},
            name="cell",
            center_ids=["a"],
            range_ids=["a"],
            check_ids=["b"],
        )
