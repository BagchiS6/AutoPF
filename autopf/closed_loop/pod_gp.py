"""A restartable, uncertainty-bearing POD--Gaussian-process field surrogate.

This implementation is dependency-light and deliberately transparent.  POD is
learned from full MOOSE fields and independent exact RBF Gaussian processes are
conditioned on the retained coefficients.  It is suitable for online campaign
sizes; larger campaigns can replace it with the publication sparse variational
model through the same strategy interface.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class PODGaussianProcess:
    """Low-rank full-field emulator with an exact shared-kernel GP posterior."""

    schema = "autopf.pod_gp.exact_rbf.v1"

    def __init__(
        self,
        *,
        max_modes: int = 32,
        explained_variance: float = 0.999,
        noise: float = 1.0e-4,
        length_scale: float | None = None,
        optimize_hyperparameters: bool = True,
    ) -> None:
        self.max_modes = int(max_modes)
        self.explained_variance = float(explained_variance)
        self.noise = float(noise)
        self.length_scale = None if length_scale is None else float(length_scale)
        self.optimize_hyperparameters = bool(optimize_hyperparameters)
        self._state: dict[str, np.ndarray | float | int] | None = None

    @property
    def fitted(self) -> bool:
        return self._state is not None

    @property
    def training_count(self) -> int:
        return 0 if self._state is None else int(np.asarray(self._state["x_raw"]).shape[0])

    @staticmethod
    def _pairwise_sq(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        delta = left[:, None, :] - right[None, :, :]
        return np.sum(delta * delta, axis=-1)

    @staticmethod
    def _kernel(left: np.ndarray, right: np.ndarray, length_scale: float) -> np.ndarray:
        return np.exp(-0.5 * PODGaussianProcess._pairwise_sq(left, right) / (length_scale * length_scale))

    def _choose_hyperparameters(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if len(x) < 2:
            return float(self.length_scale or 1.0), self.noise
        distances = np.sqrt(self._pairwise_sq(x, x))
        positive = distances[distances > 0]
        median = float(np.median(positive)) if positive.size else 1.0
        base = float(self.length_scale or max(median, 0.1))
        if not self.optimize_hyperparameters:
            return base, self.noise
        length_grid = base * np.asarray([0.35, 0.6, 1.0, 1.7, 2.8])
        noise_grid = np.asarray([self.noise * 0.1, self.noise, self.noise * 10.0])
        best = (math.inf, base, self.noise)
        for length in length_grid:
            for noise in noise_grid:
                kernel = self._kernel(x, x, float(length)) + np.eye(len(x)) * max(float(noise), 1.0e-10)
                try:
                    chol = np.linalg.cholesky(kernel)
                except np.linalg.LinAlgError:
                    continue
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
                nll = 0.5 * float(np.sum(y * alpha))
                nll += y.shape[1] * float(np.log(np.diag(chol)).sum())
                if nll < best[0]:
                    best = (nll, float(length), float(noise))
        return best[1], best[2]

    def fit(self, inputs: np.ndarray, fields: np.ndarray) -> "PODGaussianProcess":
        x_raw = np.asarray(inputs, dtype=float)
        field_shape = np.asarray(fields).shape[1:]
        field = np.asarray(fields, dtype=float).reshape((len(x_raw), -1))
        if x_raw.ndim != 2 or len(x_raw) != len(field):
            raise ValueError("inputs must be (samples, features) and align with fields.")
        if len(x_raw) < 2:
            raise ValueError("At least two MOOSE fields are required to fit a POD--GP.")
        x_mean = x_raw.mean(axis=0)
        x_scale = x_raw.std(axis=0)
        x_scale[x_scale < 1.0e-12] = 1.0
        x = (x_raw - x_mean) / x_scale
        field_mean = field.mean(axis=0)
        centered = field - field_mean
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        energy = singular * singular
        cumulative = np.cumsum(energy) / max(float(energy.sum()), np.finfo(float).eps)
        modes = int(np.searchsorted(cumulative, self.explained_variance) + 1)
        modes = max(1, min(modes, self.max_modes, len(x_raw) - 1, vt.shape[0]))
        basis = vt[:modes]
        coefficient = centered @ basis.T
        coefficient_mean = coefficient.mean(axis=0)
        coefficient_scale = coefficient.std(axis=0)
        coefficient_scale[coefficient_scale < 1.0e-12] = 1.0
        y = (coefficient - coefficient_mean) / coefficient_scale
        length, noise = self._choose_hyperparameters(x, y)
        kernel = self._kernel(x, x, length) + np.eye(len(x)) * noise
        chol = np.linalg.cholesky(kernel + np.eye(len(x)) * 1.0e-10)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        self._state = {
            "x_raw": x_raw,
            "fields_raw": field,
            "field_shape": np.asarray(field_shape, dtype=int),
            "x_mean": x_mean,
            "x_scale": x_scale,
            "field_mean": field_mean,
            "basis": basis,
            "coefficient_mean": coefficient_mean,
            "coefficient_scale": coefficient_scale,
            "x_scaled": x,
            "y_scaled": y,
            "chol": chol,
            "alpha": alpha,
            "length_scale": float(length),
            "noise": float(noise),
            "retained_modes": int(modes),
            "explained_variance_actual": float(cumulative[modes - 1]),
        }
        return self

    def update(self, inputs: np.ndarray, fields: np.ndarray) -> "PODGaussianProcess":
        new_x = np.asarray(inputs, dtype=float)
        new_fields = np.asarray(fields, dtype=float)
        if not self.fitted:
            return self.fit(new_x, new_fields)
        old_x = np.asarray(self._state["x_raw"])
        old_fields = np.asarray(self._state["fields_raw"])
        combined_x = np.concatenate((old_x, new_x), axis=0)
        combined_fields = np.concatenate((old_fields, new_fields.reshape((len(new_x), -1))), axis=0)
        # Online conditioning deliberately freezes the POD basis, transforms,
        # and kernel hyperparameters established at release.  Only the GP
        # posterior changes as real MOOSE fields arrive.
        x_scaled = (combined_x - self._state["x_mean"]) / self._state["x_scale"]
        coefficients = (combined_fields - self._state["field_mean"]) @ np.asarray(self._state["basis"]).T
        y_scaled = (
            coefficients - self._state["coefficient_mean"]
        ) / self._state["coefficient_scale"]
        length = float(self._state["length_scale"])
        noise = float(self._state["noise"])
        kernel = self._kernel(x_scaled, x_scaled, length) + np.eye(len(combined_x)) * noise
        chol = np.linalg.cholesky(kernel + np.eye(len(combined_x)) * 1.0e-10)
        self._state.update({
            "x_raw": combined_x,
            "fields_raw": combined_fields,
            "x_scaled": x_scaled,
            "y_scaled": y_scaled,
            "chol": chol,
            "alpha": np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled)),
        })
        return self

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._state is None:
            raise RuntimeError("POD--GP is not fitted.")
        x = (np.asarray(inputs, dtype=float) - self._state["x_mean"]) / self._state["x_scale"]
        train = np.asarray(self._state["x_scaled"])
        length = float(self._state["length_scale"])
        cross = self._kernel(train, x, length)
        coefficient_mean_scaled = cross.T @ np.asarray(self._state["alpha"])
        solve = np.linalg.solve(np.asarray(self._state["chol"]), cross)
        latent_variance = np.maximum(1.0 - np.sum(solve * solve, axis=0), 0.0)
        coefficient_mean = coefficient_mean_scaled * self._state["coefficient_scale"] + self._state["coefficient_mean"]
        coefficient_variance = latent_variance[:, None] * np.asarray(self._state["coefficient_scale"]) ** 2
        field_mean = np.asarray(self._state["field_mean"])[None, :] + coefficient_mean @ np.asarray(self._state["basis"])
        field_variance = coefficient_variance @ (np.asarray(self._state["basis"]) ** 2)
        shape = tuple(np.asarray(self._state["field_shape"], dtype=int))
        return field_mean.reshape((-1, *shape)), field_variance.reshape((-1, *shape))

    def thompson_fields(self, inputs: np.ndarray, *, seed: int) -> np.ndarray:
        mean, variance = self.predict(inputs)
        rng = np.random.default_rng(int(seed))
        return rng.normal(mean, np.sqrt(np.maximum(variance, 1.0e-14)))

    def receipt(self) -> dict[str, Any]:
        if self._state is None:
            return {"schema": self.schema, "fitted": False}
        return {
            "schema": self.schema,
            "fitted": True,
            "training_count": self.training_count,
            "input_dimension": int(np.asarray(self._state["x_raw"]).shape[1]),
            "field_shape": np.asarray(self._state["field_shape"], dtype=int).tolist(),
            "retained_modes": int(self._state["retained_modes"]),
            "explained_variance": float(self._state["explained_variance_actual"]),
            "length_scale": float(self._state["length_scale"]),
            "noise": float(self._state["noise"]),
            "posterior": "exact RBF GP over standardized POD coefficients",
            "pod_basis_frozen_after_initial_fit": True,
            "kernel_hyperparameters_frozen_during_online_updates": True,
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        if self._state is None:
            raise RuntimeError("Cannot save an unfitted POD--GP.")
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
        os.close(descriptor)
        try:
            payload = {key: np.asarray(value) for key, value in self._state.items()}
            payload["metadata_json"] = np.asarray(json.dumps({
                "schema": self.schema,
                "max_modes": self.max_modes,
                "explained_variance": self.explained_variance,
                "optimize_hyperparameters": self.optimize_hyperparameters,
            }))
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, destination)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "PODGaussianProcess":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema") != cls.schema:
                raise ValueError(f"Unsupported POD--GP schema: {metadata.get('schema')!r}")
            model = cls(
                max_modes=int(metadata["max_modes"]),
                explained_variance=float(metadata["explained_variance"]),
                optimize_hyperparameters=bool(metadata["optimize_hyperparameters"]),
            )
            model._state = {
                key: np.asarray(data[key]) for key in data.files if key != "metadata_json"
            }
            for key in ("length_scale", "noise", "explained_variance_actual"):
                model._state[key] = float(model._state[key])
            model._state["retained_modes"] = int(model._state["retained_modes"])
        return model
