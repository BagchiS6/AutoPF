"""Scientific objective policies for closed-loop PFM inversion.

The objective choice is explicit and serialized.  This prevents an agent or a
restart from silently changing the statistical question being optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ObjectiveMode(str, Enum):
    """Supported inversion narratives."""

    PIXEL_NMSE = "pixel_nmse"
    POSTERIOR_DISTANCE = "posterior_distance"

    @classmethod
    def parse(cls, value: str | "ObjectiveMode") -> "ObjectiveMode":
        if isinstance(value, cls):
            return value
        aliases = {
            "nmse": cls.PIXEL_NMSE,
            "residual_nmse": cls.PIXEL_NMSE,
            "ensemble": cls.POSTERIOR_DISTANCE,
            "distributional": cls.POSTERIOR_DISTANCE,
            "mahalanobis": cls.POSTERIOR_DISTANCE,
            "gaussian_nll": cls.POSTERIOR_DISTANCE,
        }
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


@dataclass(frozen=True)
class ObjectiveResult:
    scores: np.ndarray
    diagnostics: dict[str, Any]


class ObjectiveEvaluator:
    """Evaluate pixel residual or noise-aware posterior-distance objectives.

    Posterior distance is the Gaussian negative log predictive density in a
    fixed feature basis.  It contains both the Mahalanobis mismatch and the
    log-determinant uncertainty penalty.  The feature basis and experimental
    covariance must be learned from calibration data only.
    """

    def __init__(
        self,
        mode: str | ObjectiveMode,
        *,
        feature_basis: np.ndarray | None = None,
        experimental_covariance: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        variance_floor: float = 1.0e-8,
        normalize_features: bool = True,
    ) -> None:
        self.mode = ObjectiveMode.parse(mode)
        self.feature_basis = None if feature_basis is None else np.asarray(feature_basis, dtype=float)
        self.experimental_covariance = (
            None if experimental_covariance is None else np.asarray(experimental_covariance, dtype=float)
        )
        self.mask = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
        self.variance_floor = max(float(variance_floor), np.finfo(float).eps)
        self.normalize_features = bool(normalize_features)
        if self.mode is ObjectiveMode.POSTERIOR_DISTANCE:
            if self.feature_basis is None or self.experimental_covariance is None:
                raise ValueError(
                    "posterior_distance requires a calibration-only feature basis and "
                    "experimental covariance."
                )
            if self.feature_basis.ndim != 2:
                raise ValueError("feature_basis must have shape (features, pixels).")
            k = self.feature_basis.shape[0]
            if self.experimental_covariance.shape != (k, k):
                raise ValueError("experimental_covariance must have shape (features, features).")

    @classmethod
    def from_npz(
        cls,
        mode: str | ObjectiveMode,
        path: str,
        *,
        basis_key: str = "basis",
        covariance_key: str = "covariance",
        mask_key: str = "mask",
        variance_floor: float = 1.0e-8,
    ) -> "ObjectiveEvaluator":
        parsed = ObjectiveMode.parse(mode)
        if parsed is ObjectiveMode.PIXEL_NMSE:
            with np.load(path, allow_pickle=False) as data:
                mask = np.asarray(data[mask_key]) if mask_key in data.files else None
            return cls(parsed, mask=mask, variance_floor=variance_floor)
        with np.load(path, allow_pickle=False) as data:
            return cls(
                parsed,
                feature_basis=np.asarray(data[basis_key]),
                experimental_covariance=np.asarray(data[covariance_key]),
                mask=np.asarray(data[mask_key]) if mask_key in data.files else None,
                variance_floor=variance_floor,
            )

    def score(
        self,
        observation: np.ndarray,
        predictive_mean: np.ndarray,
        predictive_variance: np.ndarray | None = None,
    ) -> ObjectiveResult:
        observed = np.asarray(observation, dtype=float).reshape(-1)
        mean = np.asarray(predictive_mean, dtype=float)
        if mean.ndim == 1:
            mean = mean[None, :]
        mean = mean.reshape((mean.shape[0], -1))
        if mean.shape[1] != observed.size:
            raise ValueError(f"field size mismatch: observation={observed.size}, prediction={mean.shape[1]}")
        variance = np.zeros_like(mean) if predictive_variance is None else np.asarray(predictive_variance, dtype=float)
        if variance.ndim == 1:
            variance = variance[None, :]
        variance = np.broadcast_to(variance.reshape((variance.shape[0], -1)), mean.shape)

        use = np.ones(observed.size, dtype=bool) if self.mask is None else self.mask
        if use.size != observed.size:
            raise ValueError("objective mask does not match the field size.")
        observed_used = observed[use]
        mean_used = mean[:, use]
        variance_used = np.maximum(variance[:, use], 0.0)

        if self.mode is ObjectiveMode.PIXEL_NMSE:
            denominator = max(float(np.var(observed_used)), self.variance_floor)
            residual = mean_used - observed_used[None, :]
            scores = np.mean(residual * residual + variance_used, axis=1) / denominator
            return ObjectiveResult(
                scores=scores,
                diagnostics={
                    "mode": self.mode.value,
                    "normalization_variance": denominator,
                    "pixels": int(use.sum()),
                },
            )

        basis = self.feature_basis[:, use]
        if self.normalize_features:
            observed_centered = observed_used - float(np.mean(observed_used))
            observed_rms = max(
                float(np.sqrt(np.mean(observed_centered * observed_centered))),
                self.variance_floor,
            )
            observed_projected = observed_centered / observed_rms
            mean_centered = mean_used - np.mean(mean_used, axis=1, keepdims=True)
            mean_rms = np.maximum(
                np.sqrt(np.mean(mean_centered * mean_centered, axis=1, keepdims=True)),
                self.variance_floor,
            )
            mean_projected = mean_centered / mean_rms
            variance_used = variance_used / (mean_rms * mean_rms)
        else:
            observed_projected = observed_used
            mean_projected = mean_used
        observed_feature = basis @ observed_projected
        mean_feature = mean_projected @ basis.T
        # Diagonal field uncertainty is projected into feature covariance.
        projected_variance = variance_used @ (basis * basis).T
        mahalanobis = np.empty(mean.shape[0], dtype=float)
        logdet = np.empty(mean.shape[0], dtype=float)
        for index in range(mean.shape[0]):
            covariance = self.experimental_covariance + np.diag(projected_variance[index])
            covariance = covariance + np.eye(covariance.shape[0]) * self.variance_floor
            chol = np.linalg.cholesky(covariance)
            delta = mean_feature[index] - observed_feature
            whitened = np.linalg.solve(chol, delta)
            mahalanobis[index] = float(whitened @ whitened)
            logdet[index] = float(2.0 * np.log(np.diag(chol)).sum())
        dimension = max(self.feature_basis.shape[0], 1)
        scores = 0.5 * (mahalanobis + logdet) / dimension
        return ObjectiveResult(
            scores=scores,
            diagnostics={
                "mode": self.mode.value,
                "feature_dimension": int(dimension),
                "mahalanobis_per_feature": (mahalanobis / dimension).tolist(),
                "logdet_per_feature": (logdet / dimension).tolist(),
                "learning_score": "Gaussian NLL = Mahalanobis + log determinant",
            },
        )

    def receipt(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode.value,
            "variance_floor": self.variance_floor,
            "feature_dimension": None if self.feature_basis is None else int(self.feature_basis.shape[0]),
            "masked_pixels": None if self.mask is None else int(self.mask.sum()),
            "normalize_features": self.normalize_features,
        }
