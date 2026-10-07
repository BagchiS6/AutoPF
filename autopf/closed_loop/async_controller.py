"""File-backed per-completion POD--GP Thompson controller for MatEnsemble."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .objectives import ObjectiveEvaluator, ObjectiveMode
from .pod_gp import PODGaussianProcess
from .production import CandidateLibrary, result_field
from .schema import SimulationRequest, SimulationResult


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
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


def _lock(path: Path, timeout: float = 900.0):
    started = time.monotonic()
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            return descriptor
        except FileExistsError:
            if path.exists() and time.time() - path.stat().st_mtime > timeout:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Timed out waiting for async-controller lock {path}")
            time.sleep(0.05)


def _evaluator(context: dict[str, Any]) -> ObjectiveEvaluator:
    mode = ObjectiveMode.parse(context["objective_mode"])
    artifact = Path(context["objective_artifact"])
    if mode is ObjectiveMode.PIXEL_NMSE and artifact.stat().st_size == 0:
        return ObjectiveEvaluator(mode, variance_floor=float(context["variance_floor"]))
    return ObjectiveEvaluator.from_npz(
        mode, str(artifact), variance_floor=float(context["variance_floor"])
    )


def _candidate_input(candidate: Mapping[str, Any], context: Mapping[str, Any]) -> list[float]:
    condition = context["condition"]
    return [
        *map(float, candidate["features"]),
        *(float(condition[key]) for key in context["condition_features"]),
    ]


def _pair(candidate: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    condition = context["condition"]
    suffix = ",".join(
        f"{key}={float(condition[key]):.12g}" for key in context["condition_features"]
    )
    return f"{candidate['candidate_id']}::{suffix}"


def _normalize_result(raw: Mapping[str, Any], request: Mapping[str, Any], field_key: str) -> SimulationResult:
    payload = dict(raw)
    if "simulation_id" in payload and "iteration" in payload and "outputs" in payload:
        result = SimulationResult.from_dict(payload)
    else:
        references = dict(payload.pop("data_references", {}))
        for alias in (f"{field_key}_path", "field_path", "surface_uz_path"):
            if alias in payload and field_key not in references:
                references[field_key] = payload.pop(alias)
        outputs = dict(payload.pop("outputs", {}))
        if field_key in payload:
            outputs[field_key] = payload.pop(field_key)
        result = SimulationResult(
            simulation_id=str(payload.pop("simulation_id", request["simulation_id"])),
            iteration=int(payload.pop("iteration", request["iteration"])),
            status=str(payload.pop("status", "completed")),
            outputs=outputs,
            data_references=references,
            metadata=dict(payload.pop("metadata", {})),
        )
    return SimulationResult(
        simulation_id=result.simulation_id,
        iteration=result.iteration,
        status=result.status,
        outputs=result.outputs,
        data_references=result.data_references,
        metadata={**result.metadata, "request": dict(request)},
    )


def _initialize(context: dict[str, Any]) -> dict[str, Any]:
    initial = [dict(row) for row in context["initial_requests"]]
    return {
        "schema": "autopf.async_ts_state.v1",
        "launched": len(initial),
        "completed": 0,
        "failed": 0,
        "pending": {row["simulation_id"]: row for row in initial},
        "requests": {row["simulation_id"]: row for row in initial},
        "completed_results": [],
        "trained_simulation_ids": [],
        "launched_pairs": [row["metadata"]["pair_key"] for row in initial],
        "model_released": Path(context["model_path"]).exists(),
        "online_update_seconds": [],
        "acquisition_seconds": [],
    }


def _propose(context: dict[str, Any], state: dict[str, Any]) -> SimulationRequest | None:
    if int(state["launched"]) >= int(context["maximum_evaluations"]):
        return None
    model_path = Path(context["model_path"])
    if not model_path.exists():
        return None
    started = time.monotonic()
    library = CandidateLibrary(context["candidate_library"])
    launched = set(state["launched_pairs"])
    available = [row for row in library.rows if _pair(row, context) not in launched]
    if not available:
        return None
    inputs = np.asarray([_candidate_input(row, context) for row in available], dtype=float)
    model = PODGaussianProcess.load(model_path)
    seed = int(context["random_seed"]) + 7919 * int(state["launched"])
    sampled = model.thompson_fields(inputs, seed=seed)
    _mean, variance = model.predict(inputs)
    observation = np.load(context["observation_path"], allow_pickle=False)
    scores = _evaluator(context).score(observation, sampled, variance).scores
    index = int(np.argmin(scores))
    candidate = available[index]
    serial = int(state["launched"])
    request = SimulationRequest(
        simulation_id=f"moose-{int(context['iteration']):05d}-online-{serial:04d}-{candidate['candidate_id']}",
        iteration=int(context["iteration"]),
        parameters={
            **dict(candidate["parameters"]),
            **dict(context.get("observation_parameters", {})),
        },
        condition=dict(context["condition"]),
        metadata={
            "candidate_id": candidate["candidate_id"],
            "pair_key": _pair(candidate, context),
            "model_input": _candidate_input(candidate, context),
            "acquisition": "asynchronous posterior-aware Thompson sample from updated POD--GP",
            "sampled_objective": float(scores[index]),
            "objective_mode": context["objective_mode"],
            "async_context_path": context["context_path"],
        },
    )
    row = request.to_dict()
    state["launched"] += 1
    state["pending"][request.simulation_id] = row
    state["requests"][request.simulation_id] = row
    state["launched_pairs"].append(request.metadata["pair_key"])
    state["acquisition_seconds"].append(time.monotonic() - started)
    return request


def incorporate_and_propose(
    raw_result: Mapping[str, Any], context_path: str | os.PathLike[str]
) -> dict[str, Any] | None:
    """Condition one completed field and return its replacement request."""

    context_file = Path(context_path).resolve()
    context = _load(context_file)
    context["context_path"] = str(context_file)
    state_path = Path(context["state_path"])
    lock_path = state_path.with_suffix(".lock")
    descriptor = _lock(lock_path)
    try:
        state = _load(state_path) if state_path.exists() else _initialize(context)
        simulation_id = str(raw_result.get("simulation_id", ""))
        if not simulation_id:
            pending = list(state["pending"])
            if len(pending) != 1:
                raise ValueError("MOOSE task result must include simulation_id when several tasks are pending.")
            simulation_id = pending[0]
        if simulation_id not in state["pending"]:
            raise ValueError(f"Unexpected or duplicate simulation result {simulation_id!r}.")
        request = state["pending"].pop(simulation_id)
        result = _normalize_result(raw_result, request, context["field_key"])
        state["completed"] += 1
        if result.status != "completed":
            state["failed"] += 1
        else:
            field = result_field(result, context["field_key"])
            input_row = np.asarray(request["metadata"]["model_input"], dtype=float)[None, :]
            model_path = Path(context["model_path"])
            started = time.monotonic()
            if model_path.exists():
                PODGaussianProcess.load(model_path).update(input_row, field[None, ...]).save(model_path)
                state["trained_simulation_ids"].append(simulation_id)
                state["model_released"] = True
            else:
                completed = [*state["completed_results"], {
                    **result.to_dict(), "metadata": {**result.metadata, "request": request}
                }]
                usable = [row for row in completed if row["status"] == "completed"]
                if len(usable) >= int(context["online_release_minimum"]):
                    release_inputs = np.asarray([
                        row["metadata"]["request"]["metadata"]["model_input"] for row in usable
                    ], dtype=float)
                    release_fields = np.asarray([
                        result_field(SimulationResult.from_dict(row), context["field_key"])
                        for row in usable
                    ], dtype=float)
                    PODGaussianProcess(max_modes=int(context["max_modes"])).fit(
                        release_inputs, release_fields
                    ).save(model_path)
                    state["trained_simulation_ids"].extend(row["simulation_id"] for row in usable)
                    state["model_released"] = True
            state["online_update_seconds"].append(time.monotonic() - started)
        enriched = SimulationResult(
            simulation_id=result.simulation_id,
            iteration=result.iteration,
            status=result.status,
            outputs=result.outputs,
            data_references=result.data_references,
            metadata={
                **result.metadata,
                "request": request,
                "online_conditioned": result.status == "completed" and state["model_released"],
            },
        )
        state["completed_results"].append(enriched.to_dict())
        proposal = _propose(context, state)
        _atomic(state_path, state)
        with Path(context["results_path"]).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(enriched.to_dict(), sort_keys=True) + "\n")
        return None if proposal is None else proposal.to_dict()
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
