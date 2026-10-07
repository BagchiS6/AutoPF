"""Production POD--GP/Thompson strategy for AEcroscopyWave--MOOSE loops."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .objectives import ObjectiveEvaluator, ObjectiveMode
from .pod_gp import PODGaussianProcess
from .schema import ExperimentRequest, Observation, SimulationRequest, SimulationResult, StrategyUpdate


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def resolve_objective_mode(
    requested: str | ObjectiveMode | None,
    *,
    agent_choice_path: str | Path | None = None,
    default: str | ObjectiveMode = ObjectiveMode.POSTERIOR_DISTANCE,
) -> ObjectiveMode:
    """Resolve and freeze a human- or agent-selected objective.

    An agent hook is a small JSON decision receipt containing
    ``{"objective_mode": ...}``.  It may choose between reviewed policies but
    cannot inject executable code or alter an already checkpointed campaign.
    An explicit CLI/config choice always takes precedence.
    """

    if requested is not None:
        return ObjectiveMode.parse(requested)
    if agent_choice_path is not None:
        payload = json.loads(Path(agent_choice_path).read_text(encoding="utf-8"))
        if "objective_mode" not in payload:
            raise ValueError("Agent policy receipt must contain objective_mode.")
        return ObjectiveMode.parse(payload["objective_mode"])
    return ObjectiveMode.parse(default)


class CandidateLibrary:
    """Finite reviewed physics candidates and their surrogate coordinates."""

    def __init__(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        self.rows = [dict(row) for row in candidates]
        if not self.rows:
            raise ValueError("Candidate library is empty.")
        identifiers = [str(row["candidate_id"]) for row in self.rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("candidate_id values must be unique.")
        dimensions = {len(row.get("features", [])) for row in self.rows}
        if len(dimensions) != 1 or not next(iter(dimensions)):
            raise ValueError("Every candidate needs the same non-empty numeric features vector.")
        for row in self.rows:
            row["candidate_id"] = str(row["candidate_id"])
            row["features"] = [float(value) for value in row["features"]]
            row["parameters"] = dict(row.get("parameters", {}))

    @classmethod
    def from_json(cls, path: str | Path) -> "CandidateLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload["candidates"] if isinstance(payload, dict) else payload
        return cls(rows)

    def by_id(self, candidate_id: str) -> dict[str, Any]:
        return next(row for row in self.rows if row["candidate_id"] == candidate_id)


def _load_array_reference(reference: Any, key: str) -> np.ndarray:
    if isinstance(reference, (list, tuple)):
        return np.asarray(reference, dtype=float)
    path = Path(str(reference)).resolve()
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            selected = key if key in data.files else data.files[0]
            return np.asarray(data[selected], dtype=float)
    raise ValueError(f"Unsupported field reference {path}; expected .npy or .npz.")


def result_field(result: SimulationResult, key: str) -> np.ndarray:
    if key in result.outputs:
        return np.asarray(result.outputs[key], dtype=float)
    if key in result.data_references:
        return _load_array_reference(result.data_references[key], key)
    raise KeyError(f"Simulation {result.simulation_id} did not return {key!r}.")


class ProductionPODGPStrategy:
    """Real full-field POD--GP inversion with posterior-aware Thompson queries.

    MatEnsemble executes each proposed MOOSE batch with adaptive asynchronous
    dispatch.  At every completed campaign iteration, returned full fields are
    conditioned into the GP and the next Thompson batch is formed.  Set
    ``batch_size=1`` for strictly sequential online TS; larger values implement
    worker-filling batched TS between microscope frames.
    """

    def __init__(
        self,
        *,
        candidate_library: CandidateLibrary,
        acquisition_plan: Sequence[Mapping[str, Any]],
        model_path: str | Path,
        objective: ObjectiveEvaluator,
        condition_features: Sequence[str],
        field_key: str = "surface_uz",
        batch_size: int = 8,
        bootstrap_size: int = 16,
        random_seed: int = 20261007,
        max_modes: int = 32,
        stop_posterior_mass: float = 0.95,
        async_evaluations_per_observation: int | None = None,
        online_release_minimum: int = 8,
        async_context_root: str | Path | None = None,
        observation_parameter_map: Mapping[str, str] | None = None,
    ) -> None:
        self.library = candidate_library
        self.acquisition_plan = [dict(row) for row in acquisition_plan]
        if not self.acquisition_plan:
            raise ValueError("acquisition_plan is empty.")
        self.model_path = Path(model_path).resolve()
        self.objective = objective
        self.condition_features = tuple(str(key) for key in condition_features)
        self.field_key = str(field_key)
        self.batch_size = max(int(batch_size), 1)
        self.bootstrap_size = max(int(bootstrap_size), 2)
        self.random_seed = int(random_seed)
        self.max_modes = int(max_modes)
        self.stop_posterior_mass = float(stop_posterior_mass)
        self.async_evaluations_per_observation = (
            None if async_evaluations_per_observation is None
            else max(int(async_evaluations_per_observation), self.batch_size)
        )
        self.online_release_minimum = max(int(online_release_minimum), 2)
        self.async_context_root = Path(
            async_context_root or self.model_path.parent / "async_contexts"
        ).resolve()
        self.observation_parameter_map = dict(observation_parameter_map or {})

    def _condition(self, observation: Observation) -> dict[str, Any]:
        condition = dict(observation.metadata.get("condition", {}))
        condition.update({key: value for key, value in observation.data.items() if key in self.condition_features})
        missing = [key for key in self.condition_features if key not in condition]
        if missing:
            raise ValueError(f"Observation is missing condition features: {missing}")
        return condition

    def _input(self, candidate: Mapping[str, Any], condition: Mapping[str, Any]) -> list[float]:
        return [*map(float, candidate["features"]), *(float(condition[key]) for key in self.condition_features)]

    def _pair_key(self, candidate: Mapping[str, Any], condition: Mapping[str, Any]) -> str:
        context = ",".join(f"{key}={float(condition[key]):.12g}" for key in self.condition_features)
        return f"{candidate['candidate_id']}::{context}"

    def _already_evaluated(self, state: dict[str, Any]) -> set[str]:
        return set(state.get("strategy_state", {}).get("evaluated_pairs", []))

    def _observation_parameters(self, observation: Observation) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for source, target in self.observation_parameter_map.items():
            namespace, separator, key = source.partition(".")
            if not separator or namespace not in {"data", "data_references", "metadata"}:
                raise ValueError(
                    f"Observation parameter source {source!r} must use data.*, data_references.*, or metadata.*."
                )
            container = getattr(observation, namespace)
            if key not in container:
                raise KeyError(f"Observation does not contain {source!r} required for MOOSE.")
            values[str(target)] = container[key]
        return values

    @staticmethod
    def _diverse_indices(features: np.ndarray, count: int, seed: int) -> list[int]:
        rng = np.random.default_rng(seed)
        normalized = features.copy()
        scale = normalized.std(axis=0)
        scale[scale < 1.0e-12] = 1.0
        normalized = (normalized - normalized.mean(axis=0)) / scale
        chosen = [int(rng.integers(len(normalized)))]
        while len(chosen) < min(count, len(normalized)):
            distance = np.min(
                np.sum((normalized[:, None, :] - normalized[chosen][None, :, :]) ** 2, axis=-1),
                axis=1,
            )
            distance[chosen] = -1.0
            chosen.append(int(np.argmax(distance)))
        return chosen

    def propose_experiment(self, state: dict[str, Any], iteration: int) -> ExperimentRequest:
        plan = dict(self.acquisition_plan[iteration % len(self.acquisition_plan)])
        job_type = str(plan.pop("job_type", "pfm_image"))
        condition = {key: plan[key] for key in self.condition_features}
        return ExperimentRequest(
            request_id=f"{state['campaign_id']}-pfm-{iteration:05d}",
            iteration=iteration,
            job_type=job_type,
            payload=plan,
            metadata={"condition": condition, "objective_mode": self.objective.mode.value},
        )

    def propose_simulations(
        self, state: dict[str, Any], observation: Observation
    ) -> Sequence[SimulationRequest]:
        condition = self._condition(observation)
        evaluated = self._already_evaluated(state)
        available = [row for row in self.library.rows if self._pair_key(row, condition) not in evaluated]
        if not available:
            raise RuntimeError("No unevaluated physics candidates remain for this condition.")
        inputs = np.asarray([self._input(row, condition) for row in available], dtype=float)
        if self.model_path.exists():
            model = PODGaussianProcess.load(self.model_path)
            sampled = model.thompson_fields(inputs, seed=self.random_seed + observation.iteration)
            _mean, variance = model.predict(inputs)
            observed = np.asarray(observation.data[self.field_key], dtype=float)
            scores = self.objective.score(observed, sampled, variance).scores
            order = np.argsort(scores)[: self.batch_size]
            acquisition = "posterior-aware Thompson sample from POD--GP"
        else:
            order = self._diverse_indices(inputs, min(self.bootstrap_size, len(inputs)), self.random_seed)
            order = np.asarray(order, dtype=int)
            scores = np.full(len(available), np.nan)
            acquisition = "space-filling bootstrap before POD--GP release"
        requests = []
        observation_parameters = self._observation_parameters(observation)
        for rank, index in enumerate(order):
            candidate = available[int(index)]
            requests.append(SimulationRequest(
                simulation_id=f"moose-{observation.iteration:05d}-{rank:03d}-{candidate['candidate_id']}",
                iteration=observation.iteration,
                parameters={**dict(candidate["parameters"]), **observation_parameters},
                condition=condition,
                metadata={
                    "candidate_id": candidate["candidate_id"],
                    "pair_key": self._pair_key(candidate, condition),
                    "model_input": self._input(candidate, condition),
                    "acquisition": acquisition,
                    "sampled_objective": None if not np.isfinite(scores[int(index)]) else float(scores[int(index)]),
                    "objective_mode": self.objective.mode.value,
                },
            ))
        if self.async_evaluations_per_observation is not None:
            context_dir = self.async_context_root / f"iteration_{observation.iteration:05d}"
            context_dir.mkdir(parents=True, exist_ok=True)
            observation_path = context_dir / "observation.npy"
            np.save(observation_path, np.asarray(observation.data[self.field_key], dtype=float), allow_pickle=False)
            objective_path = context_dir / "objective.npz"
            objective_payload: dict[str, np.ndarray] = {}
            if self.objective.feature_basis is not None:
                objective_payload["basis"] = self.objective.feature_basis
            if self.objective.experimental_covariance is not None:
                objective_payload["covariance"] = self.objective.experimental_covariance
            if self.objective.mask is not None:
                objective_payload["mask"] = self.objective.mask
            np.savez_compressed(objective_path, **objective_payload)
            context_path = context_dir / "context.json"
            context = {
                "schema": "autopf.async_ts_context.v1",
                "iteration": observation.iteration,
                "field_key": self.field_key,
                "observation_path": str(observation_path),
                "objective_mode": self.objective.mode.value,
                "objective_artifact": str(objective_path),
                "variance_floor": self.objective.variance_floor,
                "condition": condition,
                "observation_parameters": observation_parameters,
                "condition_features": list(self.condition_features),
                "candidate_library": self.library.rows,
                "model_path": str(self.model_path),
                "max_modes": self.max_modes,
                "random_seed": self.random_seed + observation.iteration * 1009,
                "maximum_evaluations": self.async_evaluations_per_observation,
                "online_release_minimum": min(self.online_release_minimum, len(requests)),
                "initial_requests": [request.to_dict() for request in requests],
                "state_path": str(context_dir / "state.json"),
                "results_path": str(context_dir / "results.jsonl"),
            }
            _atomic_json(context_path, context)
            requests = [
                SimulationRequest(
                    simulation_id=request.simulation_id,
                    iteration=request.iteration,
                    parameters=request.parameters,
                    condition=request.condition,
                    metadata={**request.metadata, "async_context_path": str(context_path)},
                )
                for request in requests
            ]
        return requests

    def update(
        self,
        state: dict[str, Any],
        observation: Observation,
        simulations: Sequence[SimulationResult],
    ) -> StrategyUpdate:
        request_by_id = {
            row["simulation_id"]: row
            for row in state["iterations"][observation.iteration]["simulation_requests"]
        }
        for row in simulations:
            if row.simulation_id not in request_by_id and row.metadata.get("request"):
                request_by_id[row.simulation_id] = dict(row.metadata["request"])
        successful = [row for row in simulations if row.status == "completed"]
        if len(successful) < 2 and not self.model_path.exists():
            raise RuntimeError("Initial POD--GP release needs at least two successful MOOSE fields.")
        inputs = np.asarray([
            request_by_id[row.simulation_id]["metadata"]["model_input"] for row in successful
        ], dtype=float)
        fields = np.asarray([result_field(row, self.field_key) for row in successful], dtype=float)
        already_conditioned = bool(successful) and all(
            row.metadata.get("online_conditioned", False) for row in successful
        )
        if already_conditioned:
            model = PODGaussianProcess.load(self.model_path)
        elif self.model_path.exists():
            model = PODGaussianProcess.load(self.model_path).update(inputs, fields)
        else:
            model = PODGaussianProcess(max_modes=self.max_modes).fit(inputs, fields)
        model.save(self.model_path)

        observed = np.asarray(observation.data[self.field_key], dtype=float)
        losses = self.objective.score(observed, fields, np.zeros_like(fields))
        finite = np.asarray(losses.scores, dtype=float)
        temperature = max(float(np.median(np.abs(finite - np.median(finite)))), 1.0e-8)
        log_weight = -(finite - float(np.min(finite))) / temperature
        weights = np.exp(log_weight - float(np.max(log_weight)))
        weights /= weights.sum()
        best_index = int(np.argmin(finite))
        evaluated = self._already_evaluated(state)
        evaluated.update(request_by_id[row.simulation_id]["metadata"]["pair_key"] for row in successful)
        posterior = [
            {
                "candidate_id": request_by_id[row.simulation_id]["metadata"]["candidate_id"],
                "objective": float(finite[index]),
                "posterior_weight": float(weights[index]),
            }
            for index, row in enumerate(successful)
        ]
        strategy_state = {
            "objective_mode": self.objective.mode.value,
            "evaluated_pairs": sorted(evaluated),
            "model_path": str(self.model_path),
            "model_sha256": sha256(self.model_path),
            "model": model.receipt(),
            "latest_posterior": posterior,
            "latest_condition": self._condition(observation),
        }
        return StrategyUpdate(
            summary={
                "best_candidate_id": posterior[best_index]["candidate_id"],
                "best_objective": float(finite[best_index]),
                "maximum_batch_posterior_mass": float(np.max(weights)),
                "objective_diagnostics": losses.diagnostics,
                "successful_moose_fields": len(successful),
            },
            strategy_state=strategy_state,
            stop=bool(np.max(weights) >= self.stop_posterior_mass),
        )
