"""Check that the AttestRF package imports and its public API is available."""

from __future__ import annotations

import attestrf


def main() -> None:
    expected = {
        "action_set_from_radius",
        "continuous_interval_margin",
        "calibration_action_envelope",
        "worst_case_margin",
        "fit_reference_stratum",
        "evaluate_reference_gate",
        "identify_with_reference_check",
    }
    missing = sorted(name for name in expected if not hasattr(attestrf, name))
    if missing:
        raise SystemExit(f"missing public symbols: {', '.join(missing)}")
    print("AttestRF import check passed")


if __name__ == "__main__":
    main()
