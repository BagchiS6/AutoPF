"""Online strategy helpers for AutoPF/MatEnsemble campaigns.

The functions in this module are intentionally scheduler-light.  They prepare
candidate batches, select newly available experimental conditions, and write
cycle state.  MatEnsemble still owns task execution through ``Pipeline`` chores;
AutoPF owns the inverse-problem strategy that decides what the next chores are.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def safe_name(text: str) -> str:
    """Return a conservative label without importing the HPC execution layer."""

    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    return out.strip("_") or "item"


@dataclass
class BudgetEstimate:
    """Compact execution-budget estimate for an online cycle."""

    available_conditions: int
    candidates: int
    moose_tasks: int
    requested_nodes: int
    worker_nodes: int
    concurrent_tasks: int
    task_waves: int
    median_task_minutes: float
    p90_task_minutes: float
    estimated_walltime_hours_median: float
    estimated_walltime_hours_p90: float


def load_online_state(path: str | Path) -> dict[str, Any]:
    """Load online campaign state if it exists, otherwise return defaults."""

    state_path = Path(path)
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {
        "schema": "autopf.online_state.v1",
        "created": datetime.utcnow().isoformat(),
        "cycles": [],
        "observations": [],
    }


def write_online_state(path: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    """Persist online campaign state as JSON."""

    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = datetime.utcnow().isoformat()
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def _read_condition_manifest(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if isinstance(raw, dict):
        for key in ("available_conditions", "arrived_conditions", "condition_keys"):
            if key in raw:
                return {str(item) for item in raw[key]}
    raise ValueError(f"Unsupported condition manifest format: {path}")


def select_available_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return conditions that have arrived from the experiment.

    If no online manifest is supplied, every configured condition is treated as
    available.  This keeps offline smoke tests and paper reproduction configs
    simple while allowing a live ADM run to reveal conditions incrementally.
    """

    online = config.get("online", {})
    conditions = list(config.get("conditions", []))
    manifest = online.get("condition_manifest")
    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path(config["campaign_dir"]).resolve() / manifest_path
        available = _read_condition_manifest(manifest_path)
        return [row for row in conditions if str(row["key"]) in available]
    return [
        row
        for row in conditions
        if bool(row.get("available", row.get("arrived", True)))
    ]


def read_stage_observations(output_dir: str | Path) -> list[dict[str, Any]]:
    """Read candidate observations from all stage summary files in a DAG output."""

    observations: list[dict[str, Any]] = []
    root = Path(output_dir)
    for path in sorted(root.glob("stage_*_summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in summary.get("candidates", []):
            observations.append(
                {
                    "stage_name": summary.get("stage_name", row.get("stage_name", "")),
                    "candidate_name": row.get("candidate_name", ""),
                    "objective": float(row.get("objective", math.inf)),
                    "parameters": dict(row.get("parameters", {})),
                    "n_conditions": int(row.get("n_conditions", 0)),
                    "source": str(path),
                }
            )
    return observations


def best_observation(observations: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the finite objective observation with smallest loss."""

    finite = [
        row for row in observations
        if math.isfinite(float(row.get("objective", math.inf)))
    ]
    if not finite:
        return None
    return min(finite, key=lambda row: float(row["objective"]))


def _clip(value: float, bounds: list[float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return min(max(float(value), lo), hi)


def _candidate(name: str, parameters: dict[str, float], note: str = "") -> dict[str, Any]:
    return {"name": safe_name(name), "note": note, "parameters": parameters}


def _lhs(bounds: dict[str, list[float]], n: int, seed: int) -> list[dict[str, float]]:
    """Small dependency-free Latin hypercube sampler in bounded parameter space."""

    rng = random.Random(seed)
    keys = list(bounds)
    columns: dict[str, list[float]] = {}
    for key in keys:
        lo, hi = map(float, bounds[key])
        values = []
        for idx in range(n):
            u = (idx + rng.random()) / max(n, 1)
            values.append(lo + u * (hi - lo))
        rng.shuffle(values)
        columns[key] = values
    return [{key: columns[key][idx] for key in keys} for idx in range(n)]


def _trust_samples(
    center: dict[str, float],
    bounds: dict[str, list[float]],
    n: int,
    radius_fraction: float,
    seed: int,
) -> list[dict[str, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        row = dict(center)
        for key, interval in bounds.items():
            lo, hi = map(float, interval)
            width = max(hi - lo, 1.0e-12)
            proposal = float(center.get(key, 0.5 * (lo + hi))) + rng.gauss(0.0, radius_fraction * width)
            row[key] = _clip(proposal, interval)
        out.append(row)
    return out


def default_online_stages(
    config: dict[str, Any],
    *,
    observations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build an online candidate ladder from config, prior observations, and bounds."""

    online = config.get("online", {})
    base_input = config.get("base_input", "campaign_assets/BTO_DW_anisotropic_hidden_physics_zdecay_ic.i")
    seed = int(online.get("random_seed", 17)) + int(online.get("cycle_index", 0)) * 1009
    observations = observations or []
    best = best_observation(observations)

    baseline = list(online.get("baseline_anchors", [
        {"name": "bulk_bto_hidden_off", "parameters": {"g11": 0.5, "g12": -0.02, "g44": 0.02}},
        {"name": "dense_posterior_hidden_off", "parameters": {"g11": 0.5, "g12": -0.06, "g44": 0.02}},
    ]))

    gij_bounds = online.get("gij_bounds", {
        "g11": [0.25, 0.85],
        "g12": [-0.12, 0.02],
        "g44": [0.001, 0.08],
    })
    for idx, params in enumerate(_lhs(gij_bounds, int(online.get("gij_lhs_candidates", 6)), seed=seed)):
        baseline.append(_candidate(f"gij_lhs_{idx:02d}", params, "online Latin-hypercube gradient-energy probe"))

    best_g = {"g11": 0.5, "g12": -0.06, "g44": 0.02}
    if best is not None:
        best_g.update({k: float(v) for k, v in best.get("parameters", {}).items() if k in {"g11", "g12", "g44"}})

    scalar_candidates = [_candidate("hidden_off_reference", dict(best_g), "current best g_ij, hidden terms off")]
    for amp in online.get("screening_scalar_grid", [0.04, 0.08, 0.12]):
        row = dict(best_g)
        row["spatial_screen_amp"] = float(amp)
        scalar_candidates.append(_candidate(f"screen_amp_{float(amp):.3f}", row, "localized screening scalar probe"))
    for amp in online.get("vegard_scalar_grid", [-0.001, -0.0005, 0.0005, 0.001]):
        row = dict(best_g)
        row["spatial_vegard_amp"] = float(amp)
        scalar_candidates.append(_candidate(f"vegard_amp_{float(amp):+.4f}", row, "isotropic localized eigenstrain probe"))

    aniso_base = dict(best_g)
    aniso_base.update({"spatial_screen_amp": float(online.get("default_screen_amp", 0.08))})
    aniso_candidates = [_candidate("screening_only_control", dict(aniso_base), "screening-only control")]
    components = online.get("anisotropic_components", ["xx", "yy", "zz", "xy", "xz", "yz"])
    amplitudes = online.get("anisotropic_amplitudes", [-0.00075, 0.00075])
    for comp in components:
        for amp in amplitudes:
            row = dict(aniso_base)
            row[f"anis_vegard_{comp}_amp"] = float(amp)
            aniso_candidates.append(
                _candidate(
                    f"anis_{comp}_{float(amp):+.5f}",
                    row,
                    "one-component localized anisotropic eigenstrain probe",
                )
            )

    trust_center = dict(best.get("parameters", aniso_base) if best else aniso_base)
    trust_bounds = online.get("trust_region_bounds", {
        "spatial_screen_amp": [0.0, 0.18],
        "anis_vegard_yz_amp": [-0.0015, 0.0015],
        "anis_vegard_xz_amp": [-0.0015, 0.0015],
        "flexo_proxy_amp": [-0.1, 0.1],
    })
    trust_candidates = []
    for idx, row in enumerate(
        _trust_samples(
            trust_center,
            trust_bounds,
            int(online.get("trust_region_candidates", 8)),
            float(online.get("trust_radius_fraction", 0.20)),
            seed=seed + 29,
        )
    ):
        merged = dict(best_g)
        merged.update(row)
        trust_candidates.append(_candidate(f"trust_online_{idx:02d}", merged, "online local refinement proposal"))

    return [
        {
            "name": "baseline_gij",
            "base_input": base_input,
            "description": "Online gradient-energy anchoring.",
            "depends_on_previous_stage": False,
            "candidates": baseline,
        },
        {
            "name": "hidden_scalar",
            "base_input": base_input,
            "description": "Screening and scalar eigenstrain probes.",
            "depends_on_previous_stage": True,
            "candidates": scalar_candidates,
        },
        {
            "name": "anisotropic_hidden",
            "base_input": base_input,
            "description": "Low-rank one-component anisotropic eigenstrain probes.",
            "depends_on_previous_stage": True,
            "candidates": aniso_candidates,
        },
        {
            "name": "trust_refinement",
            "base_input": base_input,
            "description": "Bootstrap Thompson-like local refinement around the current best.",
            "depends_on_previous_stage": True,
            "candidates": trust_candidates,
        },
    ]


def estimate_cycle_budget(
    *,
    available_conditions: int,
    candidates: int,
    requested_nodes: int,
    cores_per_task: int = 64,
    cores_per_node: int = 64,
    orchestration_nodes: int = 1,
    median_task_minutes: float = 25.0,
    p90_task_minutes: float = 45.0,
) -> BudgetEstimate:
    """Estimate concurrent task waves and wallclock for a MatEnsemble cycle."""

    worker_nodes = max(int(requested_nodes) - int(orchestration_nodes), 1)
    tasks_per_node = max(int(cores_per_node) // max(int(cores_per_task), 1), 1)
    concurrent = max(worker_nodes * tasks_per_node, 1)
    tasks = int(available_conditions) * int(candidates)
    waves = int(math.ceil(tasks / concurrent)) if tasks else 0
    overhead = 0.25 + 0.03 * max(waves - 1, 0)
    return BudgetEstimate(
        available_conditions=int(available_conditions),
        candidates=int(candidates),
        moose_tasks=tasks,
        requested_nodes=int(requested_nodes),
        worker_nodes=worker_nodes,
        concurrent_tasks=concurrent,
        task_waves=waves,
        median_task_minutes=float(median_task_minutes),
        p90_task_minutes=float(p90_task_minutes),
        estimated_walltime_hours_median=waves * float(median_task_minutes) / 60.0 + overhead,
        estimated_walltime_hours_p90=waves * float(p90_task_minutes) / 60.0 + overhead,
    )


def prepare_online_config(config: dict[str, Any], output_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a cycle-ready config and metadata for the online ADM workflow."""

    cycle_config = copy.deepcopy(config)
    online = cycle_config.setdefault("online", {})
    output_path = Path(output_dir).resolve()
    state_path = Path(online.get("state_path", output_path / "online_state.json"))
    if not state_path.is_absolute():
        state_path = output_path / state_path
    state = load_online_state(state_path)

    available = select_available_conditions(cycle_config)
    if not available:
        raise ValueError("No experimental conditions are currently available for the online cycle.")
    cycle_config["conditions"] = available

    observed = list(state.get("observations", [])) + read_stage_observations(output_path)
    if online.get("autogenerate_stages", True):
        cycle_config["stages"] = default_online_stages(cycle_config, observations=observed)

    candidate_count = sum(len(stage.get("candidates", [])) for stage in cycle_config.get("stages", []))
    budget = estimate_cycle_budget(
        available_conditions=len(available),
        candidates=candidate_count,
        requested_nodes=int(online.get("requested_nodes", 128)),
        cores_per_task=int(cycle_config.get("num_cores", 64)),
        cores_per_node=int(online.get("cores_per_node", 64)),
        orchestration_nodes=int(online.get("orchestration_nodes", 1)),
        median_task_minutes=float(online.get("median_task_minutes", 25.0)),
        p90_task_minutes=float(online.get("p90_task_minutes", 45.0)),
    )
    metadata = {
        "cycle_index": int(online.get("cycle_index", len(state.get("cycles", [])))),
        "prepared": datetime.utcnow().isoformat(),
        "available_condition_keys": [row["key"] for row in available],
        "n_stages": len(cycle_config.get("stages", [])),
        "n_candidates": candidate_count,
        "budget": asdict(budget),
        "state_path": str(state_path),
    }
    cycle_dir = output_path / f"online_cycle_{metadata['cycle_index']:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    cycle_config.setdefault("online", {})["prepared_cycle"] = True
    cycle_config.setdefault("online", {})["prepared_metadata"] = metadata
    (cycle_dir / "cycle_config.json").write_text(json.dumps(cycle_config, indent=2), encoding="utf-8")
    (cycle_dir / "cycle_budget.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    state.setdefault("cycles", []).append(metadata)
    write_online_state(state_path, state)
    return cycle_config, metadata


def write_budget_table(path: str | Path, budgets: list[BudgetEstimate]) -> None:
    """Write a CSV budget table for proposal/planning documents."""

    rows = [asdict(row) for row in budgets]
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
