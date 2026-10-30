"""Exact query-specific decision boundaries for scalar nuisance actions.

The returned radius concerns the frozen model's identity decision, not label
correctness or physical coverage of the learned feature-space action.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _real_roots(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Two candidates per polynomial, using a stable quadratic formula.

    Inputs have shape [N,C]. Missing/complex roots are NaN. A polynomial that is
    identically zero is handled separately by the nominal-tie check."""
    # Equality to zero preserves the algebraic degree. A tolerance on a alone
    # can erase a valid root when the other coefficients are also very small.
    linear = a == 0
    discriminant = b.square() - 4 * a * c
    # Roundoff in a double root should not create a spurious complex pair.
    roundoff = 16 * torch.finfo(a.dtype).eps * (b.square() + (4 * a * c).abs())
    real = discriminant >= -roundoff
    square_root = discriminant.clamp_min(0).sqrt()
    q = -0.5 * (b + torch.where(b >= 0, square_root, -square_root))
    safe_a = torch.where(linear, torch.ones_like(a), a)
    safe_q = torch.where(q == 0, torch.ones_like(q), q)
    first = q / safe_a
    second = c / safe_q
    repeated_zero = (~linear) & (q == 0)
    first = torch.where(repeated_zero, -b / (2 * safe_a), first)
    second = torch.where(repeated_zero, first, second)
    nan = torch.full_like(a, float("nan"))
    first = torch.where((~linear) & real, first, nan)
    second = torch.where((~linear) & real, second, nan)
    linear_root = -c / torch.where(b == 0, torch.ones_like(b), b)
    first = torch.where(linear & (b != 0), linear_root, first)
    return torch.stack((first, second), dim=-1)


@torch.no_grad()
def exact_stability_radius(
    model,
    observed_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    center: float,
) -> dict[str, torch.Tensor]:
    """Find the nearest identity tie over all finite scalar action parameters.

    Arithmetic uses float64 on the embeddings' device without changing model
    weights. The supported rank-one family must be nonsingular for every finite
    parameter. Infinite radius means the nominal identity is unique everywhere
    within the modeled family; its boundary parameter is NaN and its competitor
    index is -1 because no finite boundary exists. A nominal tie has radius zero.
    """
    if int(getattr(model, "receiver_code_dim", 0)) != 1:
        raise ValueError("exact stability radius requires one action parameter")
    if (
        observed_embeddings.ndim != 2
        or prototypes.ndim != 2
        or observed_embeddings.shape[1] != prototypes.shape[1]
        or len(prototypes) < 1
    ):
        raise ValueError("expected compatible nonempty [N,D] and [C,D] inputs")
    if not math.isfinite(float(center)):
        raise ValueError("action center must be finite")
    device = observed_embeddings.device
    observed = observed_embeddings.to(dtype=torch.float64)
    proto = F.normalize(prototypes.to(device=device, dtype=torch.float64), dim=-1)
    if not bool(torch.isfinite(observed).all() & torch.isfinite(proto).all()):
        raise ValueError("features and prototypes must be finite")
    t0 = torch.tensor(float(center), device=device, dtype=observed.dtype).tanh()
    action = model.transport
    if hasattr(action, "bias") and hasattr(action.bias, "weight"):
        b = action.bias.weight[:, 0].to(device=device, dtype=observed.dtype)
        offset = getattr(action.bias, "bias", None)
        offset = (
            torch.zeros_like(b)
            if offset is None
            else offset.to(device=device, dtype=observed.dtype)
        )
        if not bool(torch.isfinite(b).all() & torch.isfinite(offset).all()):
            raise ValueError("action parameters must be finite")
        canonical = observed - offset - t0 * b
        rank_one = False
    elif all(hasattr(action, key) for key in ("P", "Q", "B", "rho")):
        p_matrix, q_matrix, b_matrix = action._bounded_parameters()
        p, q, b = [
            x[:, 0].to(device=device, dtype=observed.dtype)
            for x in (p_matrix, q_matrix, b_matrix)
        ]
        rho = float(action.rho)
        if not math.isfinite(rho) or not bool(
            torch.isfinite(p).all() & torch.isfinite(q).all() & torch.isfinite(b).all()
        ):
            raise ValueError("action parameters must be finite")
        k = rho * torch.dot(q, p)
        if not bool(torch.isfinite(k)) or float(k.abs()) >= 1:
            raise ValueError(
                "stability radius needs a globally nonsingular bounded action"
            )
        shifted = observed - t0 * b
        canonical = shifted - (rho * t0 / (1 + k * t0)) * (shifted @ q)[:, None] * p
        rank_one = True
    else:
        raise TypeError("unsupported nuisance action")
    scores = canonical @ proto.T
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("action produces nonfinite nominal scores")
    prediction = scores.argmax(dim=-1)
    differences = proto[prediction, None, :] - proto[None, :, :]
    competitors = (
        torch.arange(len(proto), device=device)[None, :] != prediction[:, None]
    )
    nominal_gap = scores.gather(1, prediction[:, None]) - scores
    tie = ((nominal_gap <= 0) & competitors).any(dim=1)
    if rank_one:
        x = torch.einsum("nd,ncd->nc", observed, differences)
        beta = torch.einsum("d,ncd->nc", b, differences)
        v = torch.einsum("d,ncd->nc", p, differences)
        a2 = -k * beta + rho * v * torch.dot(q, b)
        a1 = k * x - beta - rho * v * (observed @ q)[:, None]
        a0 = x
    else:
        a0 = torch.einsum("nd,ncd->nc", observed - offset, differences)
        a1 = -torch.einsum("d,ncd->nc", b, differences)
        a2 = torch.zeros_like(a0)
    if not bool(
        torch.isfinite(a0).all() & torch.isfinite(a1).all() & torch.isfinite(a2).all()
    ):
        raise ValueError("action produces nonfinite decision-boundary coefficients")
    roots = _real_roots(a2, a1, a0)
    valid = torch.isfinite(roots) & (roots > -1) & (roots < 1) & competitors[:, :, None]
    # Invalid candidates are replaced before atanh to avoid contaminating the
    # computation with values outside its domain.
    parameters = torch.atanh(torch.where(valid, roots, torch.zeros_like(roots)))
    distances = (parameters - float(center)).abs()
    distances = torch.where(valid, distances, torch.full_like(distances, float("inf")))
    radius, flat_index = distances.flatten(1).min(dim=1)
    boundary = parameters.flatten(1).gather(1, flat_index[:, None]).squeeze(1)
    boundary = torch.where(
        torch.isfinite(radius), boundary, torch.full_like(boundary, float("nan"))
    )
    radius = torch.where(tie, torch.zeros_like(radius), radius)
    boundary = torch.where(tie, torch.full_like(boundary, float(center)), boundary)
    tie_competitor = torch.where(
        competitors, nominal_gap, torch.full_like(nominal_gap, float("inf"))
    ).argmin(dim=1)
    boundary_competitor = torch.where(tie, tie_competitor, flat_index // 2)
    boundary_competitor = torch.where(
        torch.isfinite(radius),
        boundary_competitor,
        torch.full_like(boundary_competitor, -1),
    )
    return {
        "nominal_pred": prediction,
        "stability_radius": radius,
        "boundary_parameter": boundary,
        "boundary_competitor": boundary_competitor,
        "nominal_scores": scores,
    }


def normalized_stability_score(
    radius: torch.Tensor, declared_radius: float
) -> torch.Tensor:
    """Map radii to [0,1], retaining ties and the acceptance boundary at 0.5."""
    if not math.isfinite(float(declared_radius)) or declared_radius <= 0:
        raise ValueError("declared radius must be finite and strictly positive")
    if bool(torch.isnan(radius).any()) or bool((radius < 0).any()):
        raise ValueError("stability radii must be nonnegative or positive infinity")
    # 1-r/(R+r) handles R=infinity without infinity/infinity.
    return 1 - float(declared_radius) / (radius + float(declared_radius))
