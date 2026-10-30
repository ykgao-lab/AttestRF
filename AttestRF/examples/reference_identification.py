"""Run reference fitting, independent checking, and candidate identification.

The synthetic features exercise the full decision rule without RF data or
trained weights. They are a functional example, not an experimental result.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from attestrf import (
    ReferenceThresholds,
    evaluate_reference_gate,
    fit_reference_stratum,
    identify_with_reference_check,
)
from attestrf.standalone_inference import AdditiveAction


class FeatureModel(nn.Module):
    """Stand-in for a source-trained encoder's frozen feature action."""

    def __init__(self, dimension: int = 8):
        super().__init__()
        self.receiver_code_dim = 1
        self.transport = AdditiveAction(dimension)
        with torch.no_grad():
            self.transport.bias.weight.copy_(
                torch.linspace(0.1, 0.3, dimension)[:, None]
            )

    def canonicalize(self, observed: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return self.transport.inverse(observed, code)


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = FeatureModel()
    # Source prototypes are unnormalized learned vectors. This orthogonal
    # construction makes the synthetic example easy to inspect.
    prototypes = {f"device_{index}": row for index, row in enumerate(torch.eye(8))}
    reference_codes = {"device_0": 0.25, "device_1": -0.05, "device_2": 0.32}
    with torch.no_grad():
        observations = {
            identity: model.transport(
                prototypes[identity][None].repeat(24, 1),
                torch.tensor([[code]]),
            )
            for identity, code in reference_codes.items()
        }
    references = fit_reference_stratum(
        model,
        prototypes,
        observations,
        name="receiver_day",
        center_ids=["device_0"],
        range_ids=["device_1"],
        check_ids=["device_2"],
        residual_check_ids=list(reference_codes),
        steps=300,
        multistart=1,
    )
    # In a real study these limits are fixed using source recordings.
    thresholds = ReferenceThresholds(
        overall_containment=1.0,
        maximum_residual={"receiver_day": 0.01},
        minimum_accuracy={"receiver_day": 0.8},
    )
    gate = evaluate_reference_gate([references], thresholds)
    candidate_prototypes = torch.stack(
        [prototypes[f"device_{index}"] for index in range(3, 8)]
    )
    with torch.no_grad():
        observed = model.transport(candidate_prototypes, torch.tensor([[0.25]]))
    result = identify_with_reference_check(
        model, observed, candidate_prototypes, references, gate
    )
    print(f"reference_gate={gate.passed}")
    print(f"check_containment={gate.overall_containment:.3f}")
    print(f"checking_accuracy={references.checking_accuracy['device_2']:.3f}")
    print(f"reported={int(result['admitted'].sum())}/{len(observed)}")
    assert gate.passed and bool(result["admitted"].all())

    # A failed independent check withholds every candidate decision even if
    # its interval margin is positive.
    unsupported = replace(references, contained={"device_2": False})
    failed_gate = evaluate_reference_gate([unsupported], thresholds)
    withheld = identify_with_reference_check(
        model, observed, candidate_prototypes, unsupported, failed_gate
    )
    assert not bool(withheld["admitted"].any())
    print(f"reported_after_failed_reference_check={int(withheld['admitted'].sum())}")


if __name__ == "__main__":
    main()
