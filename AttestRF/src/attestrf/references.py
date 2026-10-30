"""Disjoint reference fitting and acquisition-level agreement checks.

Reference labels are authenticated independently of the RF classifier. Test
device labels are neither accepted nor used by these functions. Thresholds
must be selected using source data before inspecting a target acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .core import continuous_interval_margin
from .standalone_landmarks import (
    LandmarkActionFitter,
    LandmarkEnvelope,
    LandmarkFit,
    action_is_contained,
    calibration_action_envelope,
)


@dataclass(frozen=True)
class ReferenceStratum:
    """Reference measurements from one receiver/day or distance condition."""

    name: str
    center_ids: tuple[str, ...]
    range_ids: tuple[str, ...]
    check_ids: tuple[str, ...]
    envelope: LandmarkEnvelope
    center_fit: LandmarkFit
    individual_fits: dict[str, LandmarkFit]
    contained: dict[str, bool]
    checking_accuracy: dict[str, float]
    checked_residuals: dict[str, float]


@dataclass(frozen=True)
class ReferenceThresholds:
    """Fixed reference criteria, with optional fit and accuracy components.

    For Migration, use overall_containment=19/21 and a value of 6/7 for
    each distance in stratum_containment. For a one-reference WiSig cell,
    use the default containment threshold of one. Residual and accuracy
    thresholds are supplied separately when those components are required.
    """

    overall_containment: float = 1.0
    stratum_containment: Mapping[str, float] = field(default_factory=dict)
    maximum_residual: Mapping[str, float] = field(default_factory=dict)
    minimum_accuracy: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceGateResult:
    passed: bool
    overall_containment: float
    stratum_containment: dict[str, float]
    criteria: dict[str, bool]


@torch.no_grad()
def known_reference_accuracy(
    model,
    observed: torch.Tensor,
    identity: str,
    source_prototypes: Mapping[str, torch.Tensor],
    code: np.ndarray,
) -> float:
    """Measure a checking device against every enrolled source identity.

    ``source_prototypes`` must contain the complete source identity table,
    including reference and test-device prototypes. Restricting this table
    to reference devices would overestimate the checking accuracy.
    """

    if identity not in source_prototypes or len(source_prototypes) < 2:
        raise ValueError(
            "the reference identity and all source competitors are required"
        )
    if (
        observed.ndim != 2
        or len(observed) == 0
        or not bool(torch.isfinite(observed).all())
    ):
        raise ValueError(
            "reference observations must be a finite nonempty [N,D] tensor"
        )
    identities = list(source_prototypes)
    prototypes = torch.stack(
        [source_prototypes[key].detach() for key in identities]
    ).to(observed)
    if not bool(torch.isfinite(prototypes).all()) or bool(
        (prototypes.norm(dim=-1) == 0).any()
    ):
        raise ValueError("source prototypes must be finite and nonzero")
    head = getattr(model, "prototypes", None)
    weights = getattr(head, "weight", None)
    if weights is not None and len(weights) != len(identities):
        raise ValueError(
            "source_prototypes must include the model's complete identity table"
        )
    coordinate = torch.as_tensor(
        code, dtype=observed.dtype, device=observed.device
    ).reshape(1, -1)
    corrected = model.canonicalize(observed, coordinate.expand(len(observed), -1))
    scores = corrected @ F.normalize(prototypes, dim=-1).T
    return float((scores.argmax(dim=-1) == identities.index(identity)).float().mean())


def fit_reference_stratum(
    model,
    source_prototypes: Mapping[str, torch.Tensor],
    observations: Mapping[str, torch.Tensor],
    *,
    name: str,
    center_ids: Sequence[str],
    range_ids: Sequence[str],
    check_ids: Sequence[str],
    residual_check_ids: Sequence[str] = (),
    minimum_radius: float = 0.05,
    steps: int = 300,
    learning_rate: float = 0.05,
    multistart: int = 3,
    device: str | torch.device = "cpu",
) -> ReferenceStratum:
    """Fit the center, range, and independent checking devices separately.

    Each range/check device is fitted on its own. Checking fits never modify
    the center or the interval. The source prototype table is unnormalized
    during fitting, as required by the learned forward action.
    """

    groups = [tuple(center_ids), tuple(range_ids), tuple(check_ids)]
    flat = [identity for group in groups for identity in group]
    if any(not group for group in groups) or len(set(flat)) != len(flat):
        raise ValueError(
            "center, range, and checking roles must be nonempty and disjoint"
        )
    if not set(flat) <= set(observations) or not set(flat) <= set(source_prototypes):
        raise ValueError("every reference requires observations and a source prototype")
    if not set(residual_check_ids) <= set(flat):
        raise ValueError(
            "residual checks must refer to authenticated reference identities"
        )
    fitter = LandmarkActionFitter(
        model,
        source_prototypes,
        steps=steps,
        learning_rate=learning_rate,
        min_landmarks=1,
        trim_fraction=0.0,
        multistart=multistart,
        device=device,
    )
    center_fit = fitter.calibrate({key: observations[key] for key in groups[0]})
    individual = {
        key: fitter.calibrate({key: observations[key]}) for key in groups[1] + groups[2]
    }
    envelope = calibration_action_envelope(
        center_fit.code,
        {key: individual[key].code for key in groups[1]},
        minimum_radius,
    )
    contained = {
        key: action_is_contained(envelope, individual[key].code) for key in groups[2]
    }
    accuracy = {
        key: known_reference_accuracy(
            model,
            observations[key].to(device=fitter.device, dtype=fitter.dtype),
            key,
            source_prototypes,
            individual[key].code,
        )
        for key in groups[2]
    }
    residuals = {
        key: (
            center_fit.residuals[key]
            if key in groups[0]
            else individual[key].residuals[key]
        )
        for key in residual_check_ids
    }
    return ReferenceStratum(
        str(name),
        *groups,
        envelope,
        center_fit,
        individual,
        contained,
        accuracy,
        residuals,
    )


def evaluate_reference_gate(
    strata: Sequence[ReferenceStratum],
    thresholds: ReferenceThresholds,
) -> ReferenceGateResult:
    """Apply overall/stratum containment, residual, and accuracy criteria."""

    names = [item.name for item in strata]
    if not strata or len(set(names)) != len(names):
        raise ValueError("provide at least one stratum, with unique stratum names")
    for mapping in (
        thresholds.stratum_containment,
        thresholds.maximum_residual,
        thresholds.minimum_accuracy,
    ):
        if not set(mapping) <= set(names):
            raise ValueError("thresholds name an unavailable reference stratum")
    fractions = [
        thresholds.overall_containment,
        *thresholds.stratum_containment.values(),
        *thresholds.minimum_accuracy.values(),
    ]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("containment and accuracy thresholds must lie in [0,1]")
    if any(
        not math.isfinite(value) or value < 0
        for value in thresholds.maximum_residual.values()
    ):
        raise ValueError("residual thresholds must be finite and nonnegative")
    role_by_identity = {}
    rates = {}
    criteria = {}
    checks = []
    for item in strata:
        for role, identities in enumerate(
            (item.center_ids, item.range_ids, item.check_ids)
        ):
            for identity in identities:
                if identity in role_by_identity and role_by_identity[identity] != role:
                    raise ValueError(
                        "physical reference identities must retain disjoint roles across strata"
                    )
                role_by_identity[identity] = role
        if set(item.contained) != set(item.check_ids) or not item.check_ids:
            raise ValueError("each checking device needs a containment measurement")
        if set(item.checking_accuracy) != set(item.check_ids):
            raise ValueError(
                "each checking device needs an all-source accuracy measurement"
            )
        values = list(item.contained.values())
        checks.extend(values)
        rates[item.name] = sum(values) / len(values)
        criteria[f"containment:{item.name}"] = rates[
            item.name
        ] >= thresholds.stratum_containment.get(
            item.name,
            0.0,
        )
        if item.name in thresholds.maximum_residual:
            if not item.checked_residuals:
                raise ValueError(
                    "a residual threshold requires specified residual-check identities"
                )
            limit = thresholds.maximum_residual[item.name]
            criteria[f"residual:{item.name}"] = all(
                math.isfinite(value) and 0.0 <= value <= limit
                for value in item.checked_residuals.values()
            )
        limit = thresholds.minimum_accuracy.get(item.name, 0.0)
        criteria[f"accuracy:{item.name}"] = all(
            math.isfinite(value) and limit <= value <= 1.0
            for value in item.checking_accuracy.values()
        )
    overall = sum(checks) / len(checks)
    criteria["containment:overall"] = overall >= thresholds.overall_containment
    return ReferenceGateResult(all(criteria.values()), overall, rates, criteria)


def identify_with_reference_check(
    model,
    observed: torch.Tensor,
    candidate_prototypes: torch.Tensor,
    stratum: ReferenceStratum,
    gate: ReferenceGateResult,
) -> dict[str, torch.Tensor]:
    """Report candidate identities only when both reference and margin checks pass.

    Candidate prototypes contain only the devices being identified. The
    returned identity index is -1 when the decision is withheld. The complete
    source table used for checking references remains separate.
    """

    if stratum.name not in gate.stratum_containment:
        raise ValueError("the supplied reference gate does not contain this stratum")
    result = continuous_interval_margin(
        model,
        observed,
        candidate_prototypes,
        stratum.envelope.center,
        stratum.envelope.radius,
    )
    admitted = (result["robust_margin"] > 0.0) & gate.passed
    result["admitted"] = admitted
    result["identity"] = torch.where(admitted, result["nominal_pred"], -1)
    return result
