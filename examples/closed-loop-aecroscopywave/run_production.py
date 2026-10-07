#!/usr/bin/env python3
"""Launch live AEcroscopyWave acquisition with real AutoPF/MOOSE/POD--GP theory."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from autopf.closed_loop import ClosedLoopRunner, JsonStateStore
from autopf.closed_loop.backends import (
    AEcroscopyWaveAcquisitionBackend,
    MatEnsembleAsyncTSBackend,
    MatEnsembleMOOSEBackend,
    TiledArrayResolver,
)
from autopf.closed_loop.objectives import ObjectiveEvaluator, ObjectiveMode
from autopf.closed_loop.production import (
    CandidateLibrary,
    ProductionPODGPStrategy,
    resolve_objective_mode,
)


def expand_environment(value: Any) -> Any:
    """Expand site variables in JSON configuration without changing its schema."""

    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def load_symbol(specification: str) -> Callable[..., Any]:
    module_name, separator, symbol_name = specification.partition(":")
    if not separator:
        raise ValueError(f"Hook {specification!r} must use module:function syntax.")
    return getattr(importlib.import_module(module_name), symbol_name)


def require_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def production_components(
    config: dict[str, Any],
    objective_override: str | None,
    *,
    allow_offline_instrument: bool = False,
):
    config = expand_environment(config)
    campaign = config["campaign"]
    acquisition_config = config["acquisition"]
    simulation_config = config["simulation"]
    strategy_config = config["strategy"]

    state_path = Path(campaign["state_path"]).expanduser().resolve()
    output_root = Path(simulation_config["output_root"]).expanduser().resolve()
    scratch_root = Path(simulation_config["scratch_root"]).expanduser().resolve()
    try:
        output_root.relative_to(scratch_root)
    except ValueError as exc:
        raise ValueError(f"MOOSE output_root must be inside scratch_root: {scratch_root}") from exc
    output_root.mkdir(parents=True, exist_ok=True)

    mode = resolve_objective_mode(
        objective_override or strategy_config.get("objective_mode"),
        agent_choice_path=strategy_config.get("agent_objective_receipt"),
    )
    objective_artifact = strategy_config.get("objective_artifact")
    if mode is ObjectiveMode.POSTERIOR_DISTANCE:
        artifact = require_file(objective_artifact, "posterior-distance objective artifact")
        objective = ObjectiveEvaluator.from_npz(mode, str(artifact))
    elif objective_artifact:
        objective = ObjectiveEvaluator.from_npz(mode, str(require_file(objective_artifact, "objective artifact")))
    else:
        objective = ObjectiveEvaluator(mode)

    secret = os.environ.get(acquisition_config.get("secret_env", "AFW_AGENT_SECRET"), "")
    server_url = acquisition_config.get("server_url") or os.environ.get(
        acquisition_config.get("server_url_env", "AFW_CLOUD_SERVER_URL"), ""
    )
    instrument_id = acquisition_config.get("instrument_id") or os.environ.get(
        acquisition_config.get("instrument_id_env", "AFW_INSTRUMENT_ID"), ""
    )
    if not server_url or not instrument_id or not secret:
        if not allow_offline_instrument:
            raise ValueError("Live AEcroscopyWave server URL, instrument ID, and agent secret are required.")
        server_url = server_url or "http://offline-preflight.invalid"
        instrument_id = instrument_id or "offline-preflight"
        secret = secret or "not-a-live-secret"
    if acquisition_config.get("result_resolver"):
        resolver_factory = load_symbol(acquisition_config["result_resolver"])
        resolver = resolver_factory(acquisition_config)
    else:
        transform = load_symbol(acquisition_config["pfm_transform"])
        resolver = TiledArrayResolver(
            array_key=acquisition_config["tiled_array_key"],
            output_key=strategy_config.get("field_key", "surface_uz"),
            api_key=os.environ.get(acquisition_config.get("tiled_api_key_env", "AFW_TILED_API_KEY")),
            transform=transform,
        )
    acquisition = AEcroscopyWaveAcquisitionBackend(
        server_url=server_url,
        instrument_id=instrument_id,
        secret=secret,
        poll_interval_seconds=float(acquisition_config.get("poll_interval_seconds", 5.0)),
        timeout_seconds=float(acquisition_config.get("timeout_seconds", 7200.0)),
        result_resolver=resolver,
    )

    moose_app = require_file(simulation_config["moose_app"], "MOOSE application")
    base_input = require_file(simulation_config["base_input"], "MOOSE input")
    backend_mode = str(simulation_config.get("backend", "async_ts"))
    if backend_mode == "async_ts":
        simulation = MatEnsembleAsyncTSBackend(
            output_root=output_root,
            command_builder=simulation_config["command_builder"],
            command_config={
                **dict(simulation_config.get("command_config", {})),
                "moose_app": str(moose_app),
                "base_input": str(base_input),
                "condition_override_map": dict(simulation_config.get("condition_override_map", {})),
            },
            num_cores=int(simulation_config.get("num_cores", 64)),
            task_environment=dict(simulation_config.get("task_environment", {})),
            buffer_time=float(simulation_config.get("buffer_time", 0.0)),
            log_delay=float(simulation_config.get("log_delay", 5.0)),
        )
    elif backend_mode == "batch":
        result_loader = load_symbol(simulation_config["result_loader"])
        condition_map = dict(simulation_config.get("condition_override_map", {}))

        def arguments(request):
            values = dict(request.parameters)
            for key, value in request.condition.items():
                if key not in condition_map:
                    raise KeyError(f"No MOOSE override mapping configured for condition {key!r}.")
                values[condition_map[key]] = value
            return [f"{key}={value}" for key, value in sorted(values.items())]

        simulation = MatEnsembleMOOSEBackend(
            moose_app=moose_app,
            base_input=base_input,
            output_root=output_root,
            num_cores=int(simulation_config.get("num_cores", 64)),
            argument_builder=arguments,
            result_loader=result_loader,
            write_restart_freq=int(simulation_config.get("write_restart_freq", 1)),
            buffer_time=float(simulation_config.get("buffer_time", 0.0)),
            adaptive_load_balance=True,
        )
    else:
        raise ValueError(f"Unsupported simulation backend {backend_mode!r}.")

    candidates_path = require_file(strategy_config["candidate_library"], "candidate library")
    plan_path = require_file(strategy_config["acquisition_plan"], "acquisition plan")
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    acquisition_plan = plan_payload["acquisitions"] if isinstance(plan_payload, dict) else plan_payload
    strategy = ProductionPODGPStrategy(
        candidate_library=CandidateLibrary.from_json(candidates_path),
        acquisition_plan=acquisition_plan,
        model_path=strategy_config["model_path"],
        objective=objective,
        condition_features=strategy_config["condition_features"],
        field_key=strategy_config.get("field_key", "surface_uz"),
        batch_size=int(strategy_config.get("batch_size", 8)),
        bootstrap_size=int(strategy_config.get("bootstrap_size", 16)),
        random_seed=int(strategy_config.get("random_seed", 20261007)),
        max_modes=int(strategy_config.get("max_modes", 32)),
        stop_posterior_mass=float(strategy_config.get("stop_posterior_mass", 0.95)),
        async_evaluations_per_observation=(
            int(strategy_config["async_evaluations_per_observation"])
            if backend_mode == "async_ts" else None
        ),
        online_release_minimum=int(strategy_config.get("online_release_minimum", 8)),
        async_context_root=strategy_config.get("async_context_root"),
        observation_parameter_map=dict(strategy_config.get("observation_parameter_map", {})),
    )
    store = JsonStateStore(state_path)
    existing = store.load()
    if existing is not None:
        frozen = existing.get("metadata", {}).get("objective_mode")
        if frozen and frozen != mode.value:
            raise ValueError(f"Checkpoint froze objective_mode={frozen!r}; requested {mode.value!r}.")
    metadata = {
        "deployment": "live-aecroscopywave-real-moose",
        "objective_mode": mode.value,
        "surrogate": "online full-field POD--GP",
        "acquisition": (
            "per-completion asynchronous posterior-aware Thompson sampling"
            if backend_mode == "async_ts" else "batched posterior-aware Thompson sampling"
        ),
        "proxy_acquisition": False,
        "proxy_simulation": False,
        "scratch_root": str(scratch_root),
    }
    return store, acquisition, simulation, strategy, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--objective-mode", choices=[mode.value for mode in ObjectiveMode])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--offline-instrument-preflight",
        action="store_true",
        help="validate the theory deployment without requiring a live AFW endpoint or secrets",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    store, acquisition, simulation, strategy, metadata = production_components(
        config,
        args.objective_mode,
        allow_offline_instrument=args.offline_instrument_preflight,
    )
    if args.preflight_only:
        print(json.dumps({
            "preflight": "passed",
            "instrument_connectivity_tested": not args.offline_instrument_preflight,
            **metadata,
        }, indent=2))
        return
    runner = ClosedLoopRunner(
        campaign_id=config["campaign"]["campaign_id"],
        state_store=store,
        acquisition=acquisition,
        simulation=simulation,
        strategy=strategy,
        metadata=metadata,
    )
    state = runner.run(args.iterations)
    print(json.dumps({
        "status": state["status"],
        "next_iteration": state["next_iteration"],
        "state_path": str(store.path),
        "strategy_state": state.get("strategy_state", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
