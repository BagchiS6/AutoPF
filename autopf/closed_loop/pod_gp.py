"""Variational POD--GP field surrogate with joint BoTorch posterior sampling."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import botorch
import gpytorch
import numpy as np
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler


class _IndependentMultitaskPODGP(gpytorch.models.ApproximateGP):
    """Independent variational GP for each orthogonal POD coefficient."""

    def __init__(self, inducing_points: torch.Tensor, num_tasks: int) -> None:
        batch_shape = torch.Size([int(num_tasks)])
        if inducing_points.ndim == 2:
            inducing_points = inducing_points.unsqueeze(0).expand(
                num_tasks, -1, -1
            ).contiguous()
        distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        base = gpytorch.variational.VariationalStrategy(
            self, inducing_points, distribution, learn_inducing_locations=True
        )
        strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base, num_tasks=num_tasks
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=inducing_points.size(-1), batch_shape=batch_shape
            ),
            batch_shape=batch_shape,
        )

    def forward(self, inputs: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(inputs), self.covar_module(inputs)
        )


class PODGaussianProcess:
    """Low-rank field emulator using GPyTorch SVGPs and BoTorch joint TS.

    The POD transform is established at the initial release and then frozen.
    Each retained coefficient is modeled by an independent variational GP;
    independence is across orthogonal coefficient tasks, not across candidate
    inputs. Thompson draws are one joint posterior realization over every
    candidate and therefore retain the GP covariance across input locations.
    """

    schema = "autopf.pod_gp.gpytorch_svgp.v2"

    def __init__(
        self,
        *,
        max_modes: int = 32,
        explained_variance: float = 0.999,
        max_inducing: int = 128,
        training_steps: int = 200,
        online_steps: int = 25,
        learning_rate: float = 0.03,
        noise: float = 1.0e-4,
        device: str = "auto",
        dtype: str = "float32",
        random_seed: int = 20261007,
        **_legacy: Any,
    ) -> None:
        self.max_modes = int(max_modes)
        self.explained_variance = float(explained_variance)
        self.max_inducing = max(int(max_inducing), 2)
        self.training_steps = max(int(training_steps), 1)
        self.online_steps = max(int(online_steps), 1)
        self.learning_rate = float(learning_rate)
        self.noise = max(float(noise), 1.0e-8)
        self.device_name = str(device)
        self.dtype_name = str(dtype)
        self.random_seed = int(random_seed)
        self._state: dict[str, Any] | None = None
        self._model: _IndependentMultitaskPODGP | None = None
        self._likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood | None = None

    @property
    def _device(self) -> torch.device:
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    @property
    def _dtype(self) -> torch.dtype:
        if self.dtype_name == "float32":
            return torch.float32
        if self.dtype_name == "float64":
            return torch.float64
        raise ValueError(f"Unsupported torch dtype {self.dtype_name!r}.")

    @property
    def fitted(self) -> bool:
        return self._state is not None and self._model is not None

    @property
    def training_count(self) -> int:
        return 0 if self._state is None else int(len(self._state["x_raw"]))

    def _make_model(self, inducing: torch.Tensor, tasks: int) -> None:
        self._model = _IndependentMultitaskPODGP(inducing, tasks).to(
            device=self._device, dtype=self._dtype
        )
        self._likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=tasks, has_task_noise=False, has_global_noise=True
        ).to(device=self._device, dtype=self._dtype)
        self._likelihood.noise = self.noise

    def _train(self, x: np.ndarray, y: np.ndarray, *, steps: int, online: bool) -> list[float]:
        assert self._model is not None and self._likelihood is not None
        torch.manual_seed(self.random_seed + self.training_count)
        train_x = torch.as_tensor(x, dtype=self._dtype, device=self._device)
        train_y = torch.as_tensor(y, dtype=self._dtype, device=self._device)
        for parameter in self._model.parameters():
            parameter.requires_grad_(not online)
        for parameter in self._likelihood.parameters():
            parameter.requires_grad_(not online)
        if online:
            # Freeze inducing locations, kernel, mean, and noise; update q(u).
            for parameter in self._model.variational_parameters():
                parameter.requires_grad_(True)
        parameters = [
            parameter
            for module in (self._model, self._likelihood)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        mll = gpytorch.mlls.VariationalELBO(
            self._likelihood, self._model, num_data=len(train_x)
        )
        self._model.train()
        self._likelihood.train()
        history = []
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(self._model(train_x), train_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite variational POD--GP ELBO.")
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu()))
        self._model.eval()
        self._likelihood.eval()
        return history

    def fit(self, inputs: np.ndarray, fields: np.ndarray) -> "PODGaussianProcess":
        x_raw = np.asarray(inputs, dtype=np.float64)
        field_array = np.asarray(fields, dtype=np.float64)
        if x_raw.ndim != 2 or field_array.shape[0] != len(x_raw):
            raise ValueError("inputs must be (samples, features) and align with fields.")
        if len(x_raw) < 2:
            raise ValueError("At least two MOOSE fields are required to fit a POD--GP.")
        field_shape = field_array.shape[1:]
        flat = field_array.reshape((len(x_raw), -1))
        x_mean = x_raw.mean(axis=0)
        x_scale = x_raw.std(axis=0)
        x_scale[x_scale < 1.0e-12] = 1.0
        x_scaled = (x_raw - x_mean) / x_scale
        field_mean = flat.mean(axis=0)
        centered = flat - field_mean
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        energy = singular * singular
        cumulative = np.cumsum(energy) / max(float(energy.sum()), np.finfo(float).eps)
        modes = max(
            1,
            min(
                int(np.searchsorted(cumulative, self.explained_variance) + 1),
                self.max_modes,
                len(x_raw) - 1,
                vt.shape[0],
            ),
        )
        basis = vt[:modes]
        coefficients = centered @ basis.T
        coefficient_mean = coefficients.mean(axis=0)
        coefficient_scale = coefficients.std(axis=0)
        coefficient_scale[coefficient_scale < 1.0e-12] = 1.0
        y_scaled = (coefficients - coefficient_mean) / coefficient_scale
        generator = np.random.default_rng(self.random_seed)
        inducing_count = min(self.max_inducing, len(x_raw))
        inducing_rows = generator.choice(len(x_raw), size=inducing_count, replace=False)
        inducing = torch.as_tensor(
            x_scaled[inducing_rows], dtype=self._dtype, device=self._device
        )
        self._make_model(inducing, modes)
        self._state = {
            "x_raw": x_raw,
            "fields_raw": flat,
            "field_shape": np.asarray(field_shape, dtype=np.int64),
            "x_mean": x_mean,
            "x_scale": x_scale,
            "field_mean": field_mean,
            "basis": basis,
            "coefficient_mean": coefficient_mean,
            "coefficient_scale": coefficient_scale,
            "retained_modes": modes,
            "explained_variance_actual": float(cumulative[modes - 1]),
            "initial_elbo": [],
            "online_elbo": [],
        }
        self._state["initial_elbo"] = self._train(
            x_scaled, y_scaled, steps=self.training_steps, online=False
        )
        return self

    def _scaled_coefficients(self, flat_fields: np.ndarray) -> np.ndarray:
        assert self._state is not None
        coefficients = (
            flat_fields - self._state["field_mean"]
        ) @ np.asarray(self._state["basis"]).T
        return (
            coefficients - self._state["coefficient_mean"]
        ) / self._state["coefficient_scale"]

    def update(self, inputs: np.ndarray, fields: np.ndarray) -> "PODGaussianProcess":
        if not self.fitted:
            return self.fit(inputs, fields)
        assert self._state is not None
        new_x = np.asarray(inputs, dtype=np.float64)
        new_fields = np.asarray(fields, dtype=np.float64)
        if new_x.ndim != 2 or new_fields.shape[0] != len(new_x):
            raise ValueError("Online inputs and fields must have aligned sample dimensions.")
        flat_new = new_fields.reshape((len(new_x), -1))
        if flat_new.shape[1] != np.asarray(self._state["fields_raw"]).shape[1]:
            raise ValueError("Online field shape does not match the frozen POD basis.")
        combined_x = np.concatenate((self._state["x_raw"], new_x), axis=0)
        combined_fields = np.concatenate((self._state["fields_raw"], flat_new), axis=0)
        self._state["x_raw"] = combined_x
        self._state["fields_raw"] = combined_fields
        x_scaled = (combined_x - self._state["x_mean"]) / self._state["x_scale"]
        y_scaled = self._scaled_coefficients(combined_fields)
        history = self._train(x_scaled, y_scaled, steps=self.online_steps, online=True)
        self._state["online_elbo"] = [*self._state.get("online_elbo", []), *history]
        return self

    def _latent_distribution(self, inputs: np.ndarray):
        if not self.fitted:
            raise RuntimeError("POD--GP is not fitted.")
        assert self._state is not None and self._model is not None
        x = (np.asarray(inputs, dtype=np.float64) - self._state["x_mean"]) / self._state["x_scale"]
        tensor = torch.as_tensor(x, dtype=self._dtype, device=self._device)
        self._model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            return self._model(tensor)

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distribution = self._latent_distribution(inputs)
        assert self._state is not None
        mean_scaled = distribution.mean.detach().cpu().numpy()
        variance_scaled = distribution.variance.detach().cpu().numpy()
        coefficient_mean = (
            mean_scaled * self._state["coefficient_scale"] + self._state["coefficient_mean"]
        )
        coefficient_variance = variance_scaled * self._state["coefficient_scale"] ** 2
        field_mean = self._state["field_mean"][None, :] + coefficient_mean @ self._state["basis"]
        field_variance = coefficient_variance @ (self._state["basis"] ** 2)
        shape = tuple(np.asarray(self._state["field_shape"], dtype=int))
        return field_mean.reshape((-1, *shape)), field_variance.reshape((-1, *shape))

    def thompson_fields(
        self, inputs: np.ndarray, *, seed: int, num_samples: int = 1
    ) -> np.ndarray:
        """Draw coherent joint functions over all candidate inputs with BoTorch."""

        distribution = self._latent_distribution(inputs)
        assert self._state is not None
        posterior = GPyTorchPosterior(distribution)
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([int(num_samples)]), seed=int(seed)
        )
        with torch.no_grad():
            scaled = sampler(posterior).detach().cpu().numpy()
        if scaled.ndim != 3:
            raise RuntimeError(f"Unexpected BoTorch posterior sample shape {scaled.shape}.")
        coefficients = (
            scaled * self._state["coefficient_scale"][None, None, :]
            + self._state["coefficient_mean"][None, None, :]
        )
        fields = self._state["field_mean"][None, None, :] + coefficients @ self._state["basis"]
        shape = tuple(np.asarray(self._state["field_shape"], dtype=int))
        result = fields.reshape((int(num_samples), len(inputs), *shape))
        return result[0] if int(num_samples) == 1 else result

    def receipt(self) -> dict[str, Any]:
        if not self.fitted:
            return {"schema": self.schema, "fitted": False}
        assert self._state is not None
        return {
            "schema": self.schema,
            "fitted": True,
            "training_count": self.training_count,
            "input_dimension": int(np.asarray(self._state["x_raw"]).shape[1]),
            "field_shape": np.asarray(self._state["field_shape"], dtype=int).tolist(),
            "retained_modes": int(self._state["retained_modes"]),
            "explained_variance": float(self._state["explained_variance_actual"]),
            "surrogate": "GPyTorch independent-multitask sparse variational POD--GP",
            "posterior_sampler": "BoTorch SobolQMCNormalSampler reparameterized rsample(GPyTorchPosterior)",
            "joint_candidate_covariance_preserved": True,
            "pod_basis_frozen_after_initial_fit": True,
            "kernel_hyperparameters_frozen_during_online_updates": True,
            "online_update": "warm-start variational q(u) optimization",
            "device": str(self._device),
            "dtype": self.dtype_name,
            "torch": torch.__version__,
            "gpytorch": gpytorch.__version__,
            "botorch": botorch.__version__,
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted POD--GP.")
        assert self._state is not None and self._model is not None and self._likelihood is not None
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".pt", dir=destination.parent
        )
        os.close(descriptor)
        payload = {
            "metadata": {
                "schema": self.schema,
                "max_modes": self.max_modes,
                "explained_variance": self.explained_variance,
                "max_inducing": self.max_inducing,
                "training_steps": self.training_steps,
                "online_steps": self.online_steps,
                "learning_rate": self.learning_rate,
                "noise": self.noise,
                "dtype": self.dtype_name,
                "random_seed": self.random_seed,
            },
            "state": self._state,
            "model_state_dict": self._model.state_dict(),
            "likelihood_state_dict": self._likelihood.state_dict(),
            "inducing_points": self._model.variational_strategy.base_variational_strategy.inducing_points.detach().cpu(),
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "PODGaussianProcess":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata.get("schema") != cls.schema:
            raise ValueError(f"Unsupported POD--GP schema: {metadata.get('schema')!r}")
        model = cls(
            max_modes=metadata["max_modes"],
            explained_variance=metadata["explained_variance"],
            max_inducing=metadata["max_inducing"],
            training_steps=metadata["training_steps"],
            online_steps=metadata["online_steps"],
            learning_rate=metadata["learning_rate"],
            noise=metadata["noise"],
            dtype=metadata["dtype"],
            random_seed=metadata["random_seed"],
            device="auto",
        )
        model._state = payload["state"]
        inducing = payload["inducing_points"].to(device=model._device, dtype=model._dtype)
        model._make_model(inducing, int(model._state["retained_modes"]))
        assert model._model is not None and model._likelihood is not None
        model._model.load_state_dict(payload["model_state_dict"])
        model._likelihood.load_state_dict(payload["likelihood_state_dict"])
        model._model.eval()
        model._likelihood.eval()
        return model
