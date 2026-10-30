"""Invertibility and source-model differentiation checks."""

from __future__ import annotations

import pytest
import torch

from attestrf.standalone_inference import (
    AdditiveAction,
    IdentificationModel,
    LowRankAction,
)


@pytest.mark.parametrize("code_dim", [1, 3])
@pytest.mark.parametrize("action_type", [AdditiveAction, LowRankAction])
def test_action_round_trip(action_type, code_dim) -> None:
    torch.manual_seed(5)
    action = action_type(8, code_dim).double()
    canonical = torch.randn(9, 8, dtype=torch.float64)
    codes = torch.randn(9, code_dim, dtype=torch.float64) * 3
    recovered = action.inverse(action(canonical, codes), codes)
    torch.testing.assert_close(recovered, canonical, atol=1e-12, rtol=1e-12)


def test_source_model_forward_and_backward() -> None:
    torch.manual_seed(4)
    model = IdentificationModel(3, 2, embedding_dim=8, backbone_widths=(8, 16))
    result = model(torch.randn(4, 2, 64), torch.tensor([0, 1, 0, 1]))
    loss = torch.nn.functional.cross_entropy(
        result["logits"], torch.tensor([0, 1, 2, 0])
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.stem[0].weight.grad is not None
    assert model.receiver_codes.weight.grad is not None
    assert model.prototypes.weight.grad is not None
