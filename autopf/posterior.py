"""Finite-ensemble posterior summaries for AutoPF inverse campaigns.

AutoPF production campaigns evaluate a finite set of candidate parameter
vectors with MOOSE.  This module converts the candidate losses into a
location-resolved posterior over those candidates:

    p(theta_k | D_l) proportional to p(theta_k) exp(- beta_l L_l(theta_k)).

The resulting summaries are intentionally lightweight and JSON-safe, so they
can be attached to MatEnsemble stage outputs without adding a separate fitting
service inside the workflow.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _finite_float(value: Any, default: float = math.inf) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _location_key(metric: dict[str, Any]) -> str:
    for key in ("location_id", "dataset_location_id", "dataset_id"):
        value = metric.get(key)
        if value not in (None, ""):
            return str(value)
    return "global"


def _logsumexp(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.inf
    vmax = max(finite)
    return vmax + math.log(sum(math.exp(v - vmax) for v in finite))


def _softmax_log_weights(log_weights: list[float]) -> list[float]:
    norm = _logsumexp(log_weights)
    if not math.isfinite(norm):
        n = len(log_weights)
        return [1.0 / n for _ in log_weights] if n else []
    return [math.exp(w - norm) if math.isfinite(w) else 0.0 for w in log_weights]


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = min(max(float(q), 0.0), 1.0) * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(sorted_values[lo])
    frac = position - lo
    return float((1.0 - frac) * sorted_values[lo] + frac * sorted_values[hi])


def _numeric_parameters(candidates: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for candidate in candidates:
        for key, value in dict(candidate.get("parameters", {})).items():
            if math.isfinite(_finite_float(value)):
                keys.add(str(key))
    return sorted(keys)


def candidate_location_losses(candidates: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Return mean finite nMSE for each candidate at each location."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        name = str(candidate.get("candidate_name", candidate.get("candidate_id", "")))
        for metric in candidate.get("condition_metrics", []):
            loss = _finite_float(metric.get("surface_uz_nmse"))
            if math.isfinite(loss):
                grouped[_location_key(metric)][name].append(loss)

    out: dict[str, dict[str, float]] = {}
    for location_id, by_candidate in grouped.items():
        out[location_id] = {
            candidate_name: float(sum(losses) / len(losses))
            for candidate_name, losses in by_candidate.items()
            if losses
        }
    return out


def posterior_by_location(
    candidates: list[dict[str, Any]],
    *,
    beta: float | None = None,
    prior_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build location-resolved posterior summaries from candidate losses.

    Parameters
    ----------
    candidates:
        Candidate summaries produced by :func:`aggregate_candidate_scores`.
    beta:
        Inverse temperature applied to dimensionless nMSE.  If omitted, a
        robust value is chosen per location from the spread of candidate losses.
    prior_weights:
        Optional non-uniform candidate prior.  Missing candidates receive
        uniform prior mass.
    """

    if not candidates:
        return {"schema": "autopf.posterior.v1", "locations": {}}

    candidate_names = [str(row.get("candidate_name", row.get("candidate_id", f"cand{i}"))) for i, row in enumerate(candidates)]
    params_by_name = {name: dict(row.get("parameters", {})) for name, row in zip(candidate_names, candidates)}
    objective_by_name = {name: _finite_float(row.get("objective")) for name, row in zip(candidate_names, candidates)}
    parameter_keys = _numeric_parameters(candidates)
    losses_by_location = candidate_location_losses(candidates)
    losses_by_location["global"] = objective_by_name

    prior_weights = prior_weights or {}
    default_prior = 1.0 / max(len(candidate_names), 1)

    locations: dict[str, Any] = {}
    for location_id, losses in sorted(losses_by_location.items()):
        finite_losses = [loss for loss in losses.values() if math.isfinite(loss)]
        if not finite_losses:
            continue
        sorted_losses = sorted(finite_losses)
        if beta is None:
            if len(sorted_losses) > 1:
                scale = max(_quantile(sorted_losses, 0.75) - sorted_losses[0], 1.0e-6)
            else:
                scale = max(sorted_losses[0], 1.0e-6)
            beta_location = 1.0 / scale
        else:
            beta_location = float(beta)

        log_weights = []
        for name in candidate_names:
            loss = losses.get(name, math.inf)
            prior = max(float(prior_weights.get(name, default_prior)), 1.0e-300)
            log_weights.append(math.log(prior) - beta_location * loss if math.isfinite(loss) else -math.inf)
        weights = _softmax_log_weights(log_weights)

        posterior_mean: dict[str, float] = {}
        posterior_std: dict[str, float] = {}
        for key in parameter_keys:
            values = [_finite_float(params_by_name[name].get(key), default=0.0) for name in candidate_names]
            mean = sum(w * v for w, v in zip(weights, values))
            var = sum(w * (v - mean) ** 2 for w, v in zip(weights, values))
            posterior_mean[key] = float(mean)
            posterior_std[key] = float(math.sqrt(max(var, 0.0)))

        ranked = sorted(
            (
                {
                    "candidate_name": name,
                    "loss": losses.get(name, math.inf),
                    "posterior_weight": float(weight),
                    "parameters": params_by_name[name],
                }
                for name, weight in zip(candidate_names, weights)
                if math.isfinite(losses.get(name, math.inf))
            ),
            key=lambda row: row["posterior_weight"],
            reverse=True,
        )
        locations[location_id] = {
            "inverse_temperature_beta": float(beta_location),
            "n_candidates_with_finite_loss": len(ranked),
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "map_candidate": ranked[0] if ranked else None,
            "candidate_weights": ranked,
        }

    return {
        "schema": "autopf.posterior.v1",
        "likelihood": "p(theta_k | D_location) proportional to prior(theta_k) exp(-beta_location * mean_nMSE_location(theta_k))",
        "parameter_keys": parameter_keys,
        "locations": locations,
    }
