"""Exact interval margins and explicit action-set calculations for AttestRF."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .stability_radius import _real_roots


@dataclass(frozen=True)
class ActionSet:
    """A finite, explicit set of receiver-acquisition actions in code space."""

    center: np.ndarray
    radius: np.ndarray
    actions: np.ndarray
    construction: str = "axis_box_grid"

    def to_json(self) -> dict:
        return {
            "center": np.asarray(self.center, dtype=float).tolist(),
            "radius": np.asarray(self.radius, dtype=float).tolist(),
            "actions": np.asarray(self.actions, dtype=float).tolist(),
            "construction": self.construction,
        }


def action_set_from_radius(
    center: Sequence[float] | np.ndarray,
    radius: float | Sequence[float] | np.ndarray,
    grid_points: int = 41,
) -> ActionSet:
    """Construct a finite design inside an axis-aligned action box.

    The grid includes both endpoints and the fitted center.  For dimensions
    above three, a deterministic diagonal design is used to avoid an
    exponential number of points. A finite design is a numerical comparison,
    not a continuous-interval guarantee; use ``continuous_interval_margin``
    for the supported scalar actions.
    """

    center_arr = np.asarray(center, dtype=np.float64).reshape(-1)
    radius_arr = np.asarray(radius, dtype=np.float64)
    if radius_arr.ndim == 0:
        radius_arr = np.repeat(radius_arr, len(center_arr))
    radius_arr = radius_arr.reshape(-1)
    if len(center_arr) == 0 or len(radius_arr) != len(center_arr):
        raise ValueError("center and radius must have the same non-zero dimension")
    if (
        np.any(~np.isfinite(center_arr))
        or np.any(~np.isfinite(radius_arr))
        or np.any(radius_arr < 0)
    ):
        raise ValueError("center/radius must be finite and radius must be nonnegative")
    grid_points = int(grid_points)
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    axes = [
        np.unique(np.append(np.linspace(c - r, c + r, grid_points), c))
        for c, r in zip(center_arr, radius_arr)
    ]
    if len(axes) <= 3:
        actions = np.asarray(list(product(*axes)), dtype=np.float64)
    else:
        # Center, axis endpoints, and a deterministic set of diagonals.
        rows = [center_arr]
        for dim, axis in enumerate(axes):
            for value in (axis[0], axis[-1]):
                row = center_arr.copy()
                row[dim] = value
                rows.append(row)
        for value in np.linspace(-1.0, 1.0, grid_points):
            rows.append(center_arr + value * radius_arr)
        actions = np.unique(np.asarray(rows), axis=0)
    construction = "axis_box_grid" if len(axes) <= 3 else "axis_diagonal_design"
    return ActionSet(center_arr, radius_arr, actions, construction)


def worst_case_margin(
    model,
    observed_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    action_set: ActionSet,
    device: str | torch.device | None = None,
    chunk_size: int = 4096,
    normalize_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate identity margins over an explicit action set.

    ``normalize_embeddings=False`` returns the unnormalized linear score gap,
    which is the scale used by :func:`continuous_interval_margin`.  The sign
    is identical to a cosine gap because the per-action embedding norm is a
    positive common factor.  Set it to ``True`` only when a cosine-margin
    magnitude is specifically required; those magnitudes are not comparable
    with the analytic routine.
    """

    if observed_embeddings.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("observed_embeddings and prototypes must be [N,D] and [C,D]")
    if observed_embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("embedding dimensions differ")
    if len(action_set.actions) == 0:
        raise ValueError("action set is empty")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if prototypes.shape[0] < 2:
        raise ValueError("at least two prototypes are required")
    if (
        not bool(torch.isfinite(observed_embeddings).all())
        or not bool(torch.isfinite(prototypes).all())
        or bool((torch.linalg.vector_norm(prototypes, dim=-1) == 0).any())
    ):
        raise ValueError(
            "features must be finite and prototypes must be finite and nonzero"
        )
    target_device = (
        torch.device(device) if device is not None else observed_embeddings.device
    )
    observed = observed_embeddings.to(target_device)
    proto = F.normalize(
        prototypes.to(device=target_device, dtype=observed.dtype), dim=-1
    )
    center = torch.as_tensor(
        action_set.center, dtype=observed.dtype, device=target_device
    )
    actions = torch.as_tensor(
        action_set.actions, dtype=observed.dtype, device=target_device
    )
    with torch.no_grad():
        nominal = model.canonicalize(observed, center.expand(len(observed), -1))
        if normalize_embeddings:
            nominal = F.normalize(nominal, dim=-1)
        nominal_scores = nominal @ proto.T
        nominal_pred = nominal_scores.argmax(dim=-1)
        nominal_margin = nominal_scores.topk(k=min(2, proto.shape[0]), dim=-1).values
        nominal_margin = nominal_margin[:, 0] - (
            nominal_margin[:, 1] if proto.shape[0] > 1 else -1.0
        )
        robust_margin = torch.full(
            (len(observed),), float("inf"), dtype=observed.dtype, device=target_device
        )
        robust_best = torch.full_like(robust_margin, -float("inf"))
        for start in range(0, len(actions), max(1, int(chunk_size))):
            current = actions[start : start + chunk_size]
            tiled = (
                observed[:, None, :]
                .expand(-1, len(current), -1)
                .reshape(-1, observed.shape[1])
            )
            codes = (
                current[None, :, :]
                .expand(len(observed), -1, -1)
                .reshape(-1, current.shape[1])
            )
            canonical = model.canonicalize(tiled, codes)
            if normalize_embeddings:
                canonical = F.normalize(canonical, dim=-1)
            scores = (canonical @ proto.T).reshape(len(observed), len(current), -1)
            chosen = scores.gather(
                2, nominal_pred[:, None, None].expand(-1, len(current), 1)
            ).squeeze(-1)
            other = scores.clone()
            other.scatter_(
                2,
                nominal_pred[:, None, None].expand(-1, len(current), 1),
                -float("inf"),
            )
            margin = chosen - other.max(dim=-1).values
            robust_margin = torch.minimum(robust_margin, margin.min(dim=1).values)
            robust_best = torch.maximum(
                robust_best, scores.max(dim=2).values.max(dim=1).values
            )
    return {
        "nominal_pred": nominal_pred,
        "nominal_margin": nominal_margin,
        "robust_margin": robust_margin,
        "nominal_scores": nominal_scores,
        "robust_best_score": robust_best,
    }


def _quadratic_over_linear_minimum(
    a2: torch.Tensor,
    a1: torch.Tensor,
    a0: torch.Tensor,
    k: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Exact minimum of (a2*t^2+a1*t+a0)/(1+k*t) on [lower, upper]."""

    def value(t: torch.Tensor) -> torch.Tensor:
        return (a2 * t.square() + a1 * t + a0) / (1.0 + k * t)

    candidates = [value(lower), value(upper)]
    # Stationary points solve a2*k*t^2 + 2*a2*t + a1-k*a0 = 0.
    # A stable quadratic formula also retains small nonzero coefficients.
    roots = _real_roots(a2 * k, 2.0 * a2, a1 - k * a0)
    for root in roots.unbind(dim=-1):
        valid = torch.isfinite(root) & (root >= lower) & (root <= upper)
        safe_root = torch.where(valid, root, lower)
        candidates.append(
            torch.where(valid, value(safe_root), torch.full_like(a0, float("inf")))
        )
    return torch.stack(candidates, dim=0).min(dim=0).values


def continuous_interval_margin(
    model,
    observed_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    center: float | Sequence[float] | np.ndarray,
    radius: float | Sequence[float] | np.ndarray,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Exact worst-case linear identity margin for a one-dimensional action.

    The normalized cosine classifier and its unnormalized linear scores have
    the same argmax.  This routine minimizes the pairwise linear score gap over
    the complete continuous code interval. It supports scalar additive and
    bounded rank-one actions. Arithmetic uses float64 without changing model
    parameters, and the returned margins use the unnormalized score scale.
    A positive returned margin therefore certifies that no code in the interval
    changes the nominal identity decision, conditional on the learned action
    family and on the true code belonging to the interval.
    """

    if int(getattr(model, "receiver_code_dim", 0)) != 1:
        raise ValueError(
            "continuous analytic certificate currently requires receiver_code_dim=1"
        )
    if observed_embeddings.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("observed_embeddings and prototypes must be [N,D] and [C,D]")
    if observed_embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("embedding dimensions differ")
    if prototypes.shape[0] < 2:
        raise ValueError("at least two prototypes are required")
    target_device = (
        torch.device(device) if device is not None else observed_embeddings.device
    )
    observed = observed_embeddings.to(device=target_device, dtype=torch.float64)
    raw_proto = prototypes.to(device=target_device, dtype=torch.float64)
    if (
        not bool(torch.isfinite(observed).all())
        or not bool(torch.isfinite(raw_proto).all())
        or bool((torch.linalg.vector_norm(raw_proto, dim=-1) == 0).any())
    ):
        raise ValueError(
            "features must be finite and prototypes must be finite and nonzero"
        )
    proto = F.normalize(raw_proto, dim=-1)
    center_arr = np.asarray(center).reshape(-1)
    radius_arr = np.asarray(radius).reshape(-1)
    if center_arr.size != 1 or radius_arr.size != 1:
        raise ValueError("continuous interval center and radius must each be scalar")
    center_value = float(center_arr[0])
    radius_value = float(radius_arr[0])
    if (
        not np.isfinite(center_value)
        or not np.isfinite(radius_value)
        or radius_value < 0
    ):
        raise ValueError("center and radius must be finite, with nonnegative radius")
    c_lower = torch.as_tensor(
        center_value - radius_value, dtype=observed.dtype, device=target_device
    )
    c_upper = torch.as_tensor(
        center_value + radius_value, dtype=observed.dtype, device=target_device
    )
    t_lower, t_upper = torch.tanh(c_lower), torch.tanh(c_upper)
    t_center = torch.as_tensor(
        center_value, dtype=observed.dtype, device=target_device
    ).tanh()
    with torch.no_grad():
        action = model.transport
        if int(getattr(action, "code_dim", 1)) != 1:
            raise ValueError("the transport must have exactly one code dimension")
        if hasattr(action, "bias") and hasattr(action.bias, "weight"):
            b = action.bias.weight[:, 0].to(device=target_device, dtype=observed.dtype)
            intercept = getattr(action.bias, "bias", None)
            offset = (
                torch.zeros_like(b)
                if intercept is None
                else intercept.to(
                    device=target_device,
                    dtype=observed.dtype,
                )
            )
            nominal = observed - offset - t_center * b
            action_kind = "additive"
        elif all(hasattr(action, key) for key in ("P", "Q", "B", "rho")):
            p, q_matrix, b_matrix = action._bounded_parameters()
            u, q, b = [
                matrix[:, 0].to(device=target_device, dtype=observed.dtype)
                for matrix in (p, q_matrix, b_matrix)
            ]
            rho = torch.as_tensor(
                float(action.rho), dtype=observed.dtype, device=target_device
            )
            k = rho * torch.dot(q, u)
            denominator = torch.stack((1.0 + k * t_lower, 1.0 + k * t_upper))
            if not bool(torch.isfinite(denominator).all()) or not bool(
                (denominator > 0.0).all()
            ):
                raise ValueError("rank-one action is singular within the interval")
            shifted = observed - t_center * b
            nominal = (
                shifted
                - (rho * t_center / (1.0 + k * t_center)) * (shifted @ q)[:, None] * u
            )
            action_kind = "rank_one"
        else:
            raise TypeError(f"unsupported receiver action: {type(action).__name__}")
        if not bool(torch.isfinite(nominal).all()):
            raise ValueError("action produces nonfinite corrected features")
        nominal_scores = nominal @ proto.T
        prediction = nominal_scores.argmax(dim=-1)
        chosen_proto = proto[prediction]
        differences = chosen_proto[:, None, :] - proto[None, :, :]
        class_mask = (
            torch.arange(len(proto), device=target_device)[None, :]
            != prediction[:, None]
        )
        if action_kind == "additive":
            # For an affine additive action, canonicalization subtracts the
            # fixed intercept as well as the code-dependent term.
            base = torch.einsum("nd,ncd->nc", observed - offset, differences)
            slope = torch.einsum("d,ncd->nc", b, differences)
            lower_values = base - t_lower * slope
            upper_values = base - t_upper * slope
            pairwise = torch.minimum(lower_values, upper_values)
        else:
            xv = torch.einsum("nd,ncd->nc", observed, differences)
            bv = torch.einsum("d,ncd->nc", b, differences)
            uv = torch.einsum("d,ncd->nc", u, differences)
            qx = observed @ q
            qb = torch.dot(q, b)
            a0 = xv
            a1 = xv * k - bv - uv * rho * qx[:, None]
            a2 = -bv * k + uv * rho * qb
            pairwise = _quadratic_over_linear_minimum(a2, a1, a0, k, t_lower, t_upper)
        pairwise = torch.where(
            class_mask, pairwise, torch.full_like(pairwise, float("inf"))
        )
        robust_margin, worst_competitor = pairwise.min(dim=1)
        nominal_top = nominal_scores.topk(k=min(2, len(proto)), dim=-1).values
        nominal_margin = nominal_top[:, 0] - (
            nominal_top[:, 1] if len(proto) > 1 else -1.0
        )
    return {
        "nominal_pred": prediction,
        "nominal_margin": nominal_margin,
        "robust_margin": robust_margin,
        "worst_competitor": worst_competitor,
        "nominal_scores": nominal_scores,
    }
