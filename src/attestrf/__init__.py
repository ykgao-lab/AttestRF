"""AttestRF: reference-guided stability margins for RF fingerprint identification."""

from .core import (
    ActionSet,
    action_set_from_radius,
    continuous_interval_margin,
    worst_case_margin,
)
from .standalone_landmarks import (
    LandmarkActionFitter,
    LandmarkEnvelope,
    action_is_contained,
    calibration_action_envelope,
)
from .references import (
    ReferenceThresholds,
    evaluate_reference_gate,
    fit_reference_stratum,
    identify_with_reference_check,
    known_reference_accuracy,
)

__all__ = [
    "ActionSet",
    "action_set_from_radius",
    "continuous_interval_margin",
    "worst_case_margin",
    "LandmarkEnvelope",
    "LandmarkActionFitter",
    "action_is_contained",
    "calibration_action_envelope",
    "ReferenceThresholds",
    "evaluate_reference_gate",
    "fit_reference_stratum",
    "identify_with_reference_check",
    "known_reference_accuracy",
]
