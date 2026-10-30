"""Landmark-only fitting of a frozen receiver action.

The model weights remain fixed while the receiver action is fitted from
authenticated landmark observations.  The envelope reports empirical action
spread and does not by itself provide a population or Byzantine guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class LandmarkFit:
    code: np.ndarray
    landmark_ids: list[str]
    residuals: dict[str, float]
    retained_landmarks: list[str]
    q_fit: float
    q_cond: float
    q_loo: float
    optimizer_steps: int


@dataclass(frozen=True)
class LandmarkEnvelope:
    """Finite-sample action envelope defined by disjoint calibration identities."""

    center: np.ndarray
    radius: np.ndarray
    calibration_codes: dict[str, np.ndarray]
    calibration_deviations: dict[str, np.ndarray]
    minimum_radius: float


def calibration_action_envelope(
    center: np.ndarray,
    calibration_codes: Mapping[str, np.ndarray],
    minimum_radius: float = 0.05,
) -> LandmarkEnvelope:
    """Form a coordinatewise envelope from held-out calibration-identity actions.

    This construction measures cross-identity action spread.  It deliberately
    differs from leave-one-out movement of the pooled center, which measures
    estimator sensitivity and need not cover another identity's fitted action.
    No confidence or population-coverage interpretation is implied.
    """

    center_array = np.asarray(center, dtype=np.float64).reshape(-1)
    if not len(center_array) or not np.isfinite(center_array).all():
        raise ValueError("center must be a finite nonempty vector")
    if not np.isfinite(minimum_radius) or float(minimum_radius) < 0.0:
        raise ValueError("minimum_radius must be finite and nonnegative")
    if not calibration_codes:
        raise ValueError("at least one calibration identity is required")
    codes: dict[str, np.ndarray] = {}
    deviations: dict[str, np.ndarray] = {}
    for identity, value in calibration_codes.items():
        code = np.asarray(value, dtype=np.float64).reshape(-1)
        if code.shape != center_array.shape or not np.isfinite(code).all():
            raise ValueError(
                "calibration codes must be finite and match the center dimension"
            )
        key = str(identity)
        if key in codes:
            raise ValueError(f"duplicate calibration identity: {key}")
        codes[key] = code.copy()
        deviations[key] = np.abs(code - center_array)
    radius = np.maximum(
        float(minimum_radius),
        np.stack(list(deviations.values()), axis=0).max(axis=0),
    )
    return LandmarkEnvelope(
        center=center_array.copy(),
        radius=radius,
        calibration_codes=codes,
        calibration_deviations=deviations,
        minimum_radius=float(minimum_radius),
    )


def action_is_contained(
    envelope: LandmarkEnvelope,
    code: np.ndarray,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether a checking action lies in the closed fitted envelope."""

    value = np.asarray(code, dtype=np.float64).reshape(-1)
    if value.shape != envelope.center.shape or not np.isfinite(value).all():
        raise ValueError(
            "checking code must be finite and match the envelope dimension"
        )
    if not np.isfinite(tolerance) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    return bool(
        np.all(np.abs(value - envelope.center) <= envelope.radius + float(tolerance))
    )


class LandmarkActionFitter:
    """Optimize only a deployment code, leaving all model weights unchanged."""

    def __init__(
        self,
        model,
        prototypes: Mapping[str, torch.Tensor],
        steps: int = 300,
        learning_rate: float = 0.05,
        trim_fraction: float = 0.2,
        min_landmarks: int | None = None,
        huber_delta: float = 1.0,
        loss: str = "huber",
        multistart: int = 3,
        device: str | torch.device = "cpu",
    ):
        if steps < 1 or learning_rate <= 0.0 or huber_delta <= 0.0:
            raise ValueError("steps, learning rate, and Huber delta must be positive")
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError("trim fraction must lie in [0, 1)")
        if loss not in {"huber", "l2"} or multistart < 1:
            raise ValueError(
                "loss must be 'huber' or 'l2' and multistart must be positive"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.dtype = next(self.model.transport.parameters()).dtype
        self.prototypes = {
            str(k): v.detach().to(device=self.device, dtype=self.dtype)
            for k, v in prototypes.items()
        }
        if not self.prototypes or any(
            value.ndim != 1
            or not bool(torch.isfinite(value).all())
            or float(value.norm()) == 0.0
            for value in self.prototypes.values()
        ):
            raise ValueError("prototypes must be finite nonzero feature vectors")
        if len({value.shape for value in self.prototypes.values()}) != 1:
            raise ValueError("prototype feature dimensions must agree")
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.trim_fraction = float(trim_fraction)
        self.min_landmarks = int(
            model.receiver_code_dim + 1 if min_landmarks is None else min_landmarks
        )
        if self.min_landmarks < 1:
            raise ValueError("min_landmarks must be positive")
        self.huber_delta = float(huber_delta)
        self.loss = loss
        self.multistart = int(multistart)

    def _fit(self, means, ids, initial):
        code = torch.nn.Parameter(
            initial.detach().clone().to(device=self.device, dtype=self.dtype)
        )
        optimizer = torch.optim.Adam([code], lr=self.learning_rate)
        prototypes = torch.stack([self.prototypes[key] for key in ids])
        observations = torch.stack([means[key] for key in ids])
        for _ in range(self.steps):
            predicted = self.model.transport(
                prototypes, code.unsqueeze(0).expand(len(ids), -1)
            )
            residuals = torch.linalg.vector_norm(
                F.normalize(predicted, dim=-1) - observations, dim=-1
            )
            objective = (
                F.huber_loss(
                    residuals, torch.zeros_like(residuals), delta=self.huber_delta
                )
                if self.loss == "huber"
                else 0.5 * residuals.square().mean()
            )
            # Differentiate only the deployment code.  This avoids accumulating
            # unused gradients in the frozen encoder and action weights.
            (gradient,) = torch.autograd.grad(objective, code)
            optimizer.zero_grad(set_to_none=True)
            code.grad = gradient
            optimizer.step()
        with torch.no_grad():
            predicted = self.model.transport(
                prototypes, code.unsqueeze(0).expand(len(ids), -1)
            )
            residuals = torch.linalg.vector_norm(
                F.normalize(predicted, dim=-1) - observations, dim=-1
            )
        return code.detach(), residuals.detach()

    def _condition(self, ids, code):
        prototypes = torch.stack([self.prototypes[key] for key in ids])

        def prediction(current):
            return self.model.transport(
                prototypes, current.unsqueeze(0).expand(len(ids), -1)
            ).reshape(-1)

        jacobian = torch.autograd.functional.jacobian(prediction, code).reshape(
            -1, code.numel()
        )
        singular = np.linalg.svd(jacobian.detach().cpu().numpy(), compute_uv=False)
        return (
            float(singular.max() / max(float(singular.min()), 1e-8))
            if len(singular)
            else float("inf")
        )

    def calibrate(self, observations: Mapping[str, torch.Tensor]) -> LandmarkFit:
        ids = [key for key in self.prototypes if key in observations]
        if len(ids) < self.min_landmarks:
            raise ValueError(
                f"need at least {self.min_landmarks} landmarks, got {len(ids)}"
            )
        means = {}
        for key in ids:
            value = observations[key].detach().to(device=self.device, dtype=self.dtype)
            if (
                value.ndim != 2
                or len(value) == 0
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    "landmark observations must be finite, nonempty [packets, features] arrays"
                )
            if (
                value.shape[1] != self.prototypes[key].numel()
                or float(value.mean(dim=0).norm()) == 0.0
            ):
                raise ValueError(
                    "landmark centroids must be nonzero and match prototype dimensions"
                )
            means[key] = F.normalize(value.mean(dim=0), dim=0)
        candidates = []
        for index in range(self.multistart):
            initial = torch.zeros(
                self.model.receiver_code_dim, device=self.device, dtype=self.dtype
            )
            if index:
                initial = torch.randn_like(initial) * 0.1
            candidates.append(self._fit(means, ids, initial))
        code, residuals = min(
            candidates, key=lambda candidate: float(candidate[1].sum())
        )
        count = max(
            self.min_landmarks, int(np.ceil(len(ids) * (1.0 - self.trim_fraction)))
        )
        retained = [ids[index] for index in residuals.argsort().cpu().tolist()[:count]]
        if len(retained) < len(ids):
            code, residuals = self._fit(means, retained, code)
            residual_ids = retained
        else:
            residual_ids = ids
        residual_map = {
            key: float(value)
            for key, value in zip(residual_ids, residuals.cpu().tolist())
        }
        with torch.no_grad():
            for key in ids:
                if key not in residual_map:
                    predicted = self.model.transport(
                        self.prototypes[key].unsqueeze(0), code.unsqueeze(0)
                    )
                    residual_map[key] = float(
                        torch.linalg.vector_norm(
                            F.normalize(predicted, dim=-1)[0] - means[key]
                        )
                    )
        drifts = []
        for omitted in retained:
            subset = [key for key in retained if key != omitted]
            if len(subset) >= self.min_landmarks:
                leave_code, _ = self._fit(means, subset, code)
                drifts.append(float(torch.linalg.vector_norm(leave_code - code).cpu()))
        return LandmarkFit(
            code=code.cpu().numpy(),
            landmark_ids=ids,
            residuals=residual_map,
            retained_landmarks=retained,
            q_fit=float(np.median(list(residual_map.values()))),
            q_cond=self._condition(retained, code),
            q_loo=max(drifts, default=0.0),
            optimizer_steps=self.steps,
        )
