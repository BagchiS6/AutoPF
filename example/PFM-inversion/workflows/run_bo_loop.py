#!/usr/bin/env python3
"""Autonomous simulation-only BO loop for phase-field calibration.

Runs entirely on NERSC after a one-time staging step from the laptop.
No live experimental data needed — the PFM reference images are staged
once as .npz + polar_z files.

Each round:
  1. Select candidate (g11, g12, g44) vectors via GP Thompson sampling
     (or random on round 0 / when BoTorch is unavailable)
  2. Build a per-round run manifest (one MOOSE job per candidate × PFM condition)
  3. Execute in parallel via MatEnsemble / automoose
  4. Post-process Exodus outputs → fields_final_timestep.npz
  5. Compute surface_uz MSE against each staged PFM reference
  6. Update bo_state.json for the next round

Usage (sourced from perlmutter_sim_bo.sh):
  python run_bo_loop.py --config sim_bo_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from surface_orientation import orient_sim_for_experiment, resample_array, surface_uz_orientation


# ---------------------------------------------------------------------------
# Config / state helpers
# ---------------------------------------------------------------------------

CAMPAIGN_DIR = Path(os.environ.get("SIM_BO_CAMPAIGN_DIR", Path(__file__).parent))


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    # Paths in the config are already absolute (rewritten at staging time)
    return cfg


def _load_bo_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"observations": [], "objective_name": "surface_uz_nmse", "maximize": False}


def _save_bo_state(state: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parameter space — continuous Sobol sampling (dimension-agnostic)
# ---------------------------------------------------------------------------

def _param_keys(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["parameter_space"].keys())


def _param_bounds(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ps = cfg["parameter_space"]
    keys = _param_keys(cfg)
    lo = np.array([ps[k]["lower"] for k in keys], dtype=float)
    hi = np.array([ps[k]["upper"] for k in keys], dtype=float)
    return lo, hi


def _normalize(raw: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    lo, hi = _param_bounds(cfg)
    return (raw - lo) / (hi - lo)


def _denormalize(unit: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    lo, hi = _param_bounds(cfg)
    return unit * (hi - lo) + lo


def _sample_candidates(
    cfg: dict[str, Any],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-random Sobol sample of parameter space.

    Returns (raw, unit) each shaped (n_samples, n_params).
    Falls back to uniform random if scipy is unavailable.
    """
    d = len(_param_keys(cfg))
    seed = int(rng.integers(1 << 31))
    try:
        from scipy.stats.qmc import Sobol
        unit = Sobol(d=d, scramble=True, seed=seed).random(n_samples)
    except ImportError:
        unit = rng.uniform(0.0, 1.0, (n_samples, d))
    return _denormalize(unit, cfg), unit


def _array_to_candidate(row: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    return {k: float(row[i]) for i, k in enumerate(_param_keys(cfg))}


def _build_train_arrays(
    bo_state: dict[str, Any], cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_x_normalized, train_y) from bo_state observations."""
    keys = _param_keys(cfg)
    obs = bo_state["observations"]
    raw = np.array([[o["parameters"][k] for k in keys] for o in obs], dtype=float)
    y   = np.array([o["objective"] for o in obs], dtype=float)
    return _normalize(raw, cfg), y


# ---------------------------------------------------------------------------
# Candidate selection — BoTorch TS with numpy fallback
# ---------------------------------------------------------------------------

def _thompson_sample_botorch(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate_grid_norm: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Return indices into candidate_grid_norm via pathwise Thompson sampling.

    Draws batch_size independent GP posterior sample paths using Matheron's
    rule (draw_matheron_paths), evaluates each path over all candidates
    jointly, and returns the argmax index per path.  Joint evaluation means
    every candidate is scored on the same GP realization — unlike marginal
    rsample which evaluates candidates independently.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.sampling.pathwise import draw_matheron_paths
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dtype = torch.double
    X = torch.tensor(train_x, dtype=dtype)
    Y = -torch.tensor(train_y, dtype=dtype).unsqueeze(-1)  # negate: minimize MSE

    model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    X_cand = torch.tensor(candidate_grid_norm, dtype=dtype)  # [n_cand, d]

    with torch.no_grad():
        paths = draw_matheron_paths(model, sample_shape=torch.Size([batch_size]))
        # paths(X_cand) → [batch_size, n_cand, 1]
        values = paths(X_cand).squeeze(-1)   # [batch_size, n_cand]
        best_idx = values.argmax(dim=-1)     # [batch_size]

    return best_idx.cpu().numpy().astype(int)


def _rbf_gp_thompson_sample(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate_grid_norm: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Diagonal normal draw from a hand-rolled RBF GP (numpy-only fallback)."""
    n, d = train_x.shape
    ls = np.full(d, 0.25)

    def _rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = a / ls
        b2 = b / ls
        diff = a2[:, None, :] - b2[None, :, :]
        return np.exp(-0.5 * np.sum(diff ** 2, axis=2))

    y_mean = float(np.mean(train_y))
    y_std = float(np.std(train_y)) or 1.0
    y_scaled = (train_y - y_mean) / y_std

    K = _rbf(train_x, train_x) + 1e-6 * np.eye(n)
    chol = np.linalg.cholesky(K + 1e-9 * np.eye(n))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_scaled))

    k_star = _rbf(train_x, candidate_grid_norm)
    mu = k_star.T @ alpha
    v = np.linalg.solve(chol, k_star)
    var = np.maximum(1.0 - np.sum(v ** 2, axis=0), 0.0)

    sampled = rng.normal(mu * y_std + y_mean, np.sqrt(var) * y_std)
    scores = -sampled  # minimize MSE → negate for argmax selection

    order = np.argsort(scores)[::-1]
    return order[:batch_size].astype(int)


def select_candidates(
    bo_state: dict[str, Any],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    batch_size = cfg["batch_size"]
    n_samples  = cfg.get("num_candidate_samples", 8192)

    raw, unit = _sample_candidates(cfg, n_samples=n_samples, rng=rng)

    if not bo_state["observations"]:
        chosen = rng.choice(len(raw), size=batch_size, replace=False)
        return [_array_to_candidate(raw[i], cfg) for i in chosen]

    train_x, train_y = _build_train_arrays(bo_state, cfg)

    try:
        indices = _thompson_sample_botorch(train_x, train_y, unit, batch_size=batch_size)
        print("  [BO] BoTorch pathwise Thompson sampling used (draw_matheron_paths).")
    except ImportError:
        print("  [BO] BoTorch not found — falling back to numpy RBF GP.")
        indices = _rbf_gp_thompson_sample(train_x, train_y, unit, batch_size=batch_size, rng=rng)
    except Exception as exc:
        print(f"  [BO] BoTorch failed ({exc!r}) — falling back to numpy RBF GP.")
        indices = _rbf_gp_thompson_sample(train_x, train_y, unit, batch_size=batch_size, rng=rng)

    return [_array_to_candidate(raw[i], cfg) for i in indices]


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_round_manifest(
    round_idx: int | str,
    candidates: list[dict[str, float]],
    cfg: dict[str, Any],
    campaign_dir: Path,
    *,
    ref_key: str = "train_refs",
) -> dict[str, Any]:
    experiments = cfg[ref_key]
    round_dir = campaign_dir / (f"round_{round_idx:03d}" if isinstance(round_idx, int) else f"round_{round_idx}")
    runs_dir = round_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for ci, cand in enumerate(candidates):
        cand_id = f"r{round_idx:03d}_cand{ci:02d}" if isinstance(round_idx, int) else f"r{round_idx}_cand{ci:02d}"
        for exp in experiments:
            run_key = f"{cand_id}__{exp['key']}"
            output_dir = runs_dir / f"cand{ci:02d}" / exp["key"]
            output_dir.mkdir(parents=True, exist_ok=True)
            runs.append({
                "key": run_key,
                "candidate_id": cand_id,
                "experiment_key": exp["key"],
                "params": dict(cand),          # all BO parameters, dimension-agnostic
                "tip_voltage": exp["tip_voltage"],
                "pulse_end": exp["pulse_end"],
                "pfm_image_file": exp["polar_z_path"],
                "experimental_uz_path": exp["npz_path"],
                "output_dir": str(output_dir),
            })
            if "free_energy_final_reference" in exp:
                runs[-1]["free_energy_final_reference"] = exp["free_energy_final_reference"]
            if cfg.get("initial_condition", {}).get("enabled", False):
                from initial_condition_geometry import maybe_add_ic_geometry
                maybe_add_ic_geometry(runs[-1], cfg)

    manifest = {
        "schema": "aecroscopy.bo_manifest.v1",
        "round": round_idx,
        "created": datetime.utcnow().isoformat(),
        "base_input": cfg["base_input"],
        "executable": cfg["executable"],
        "num_cores": cfg["num_cores"],
        "postprocess_script": cfg["postprocess_script"],
        "loss": cfg.get("loss", {}),
        "runs": runs,
    }
    manifest_path = round_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# MOOSE execution via MatEnsemble
# ---------------------------------------------------------------------------

def run_round(manifest: dict[str, Any]) -> None:
    from autopf.utils import automoose, postproc

    runs = manifest["runs"]

    def _has_postprocessed_output(run: dict[str, Any]) -> bool:
        return (Path(run["output_dir"]) / "fields_final_timestep.npz").exists()

    def _has_finished_moose_output(run: dict[str, Any]) -> bool:
        out_dir = Path(run["output_dir"])
        if not any(out_dir.glob("*.e")):
            return False
        stdout_path = out_dir / "stdout"
        if not stdout_path.exists():
            return False
        try:
            text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return "Finished Executing" in text

    skip_completed = os.environ.get("SIM_BO_SKIP_COMPLETED_RUNS", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if skip_completed:
        pending_runs = [
            r for r in runs
            if not _has_postprocessed_output(r)
        ]
        skipped = len(runs) - len(pending_runs)
        if skipped:
            print(f"  Skipping {skipped} runs with existing fields_final_timestep.npz.")
    else:
        pending_runs = runs

    if not pending_runs:
        print("  All runs already have post-processed outputs; skipping MOOSE launch.")
        return

    if skip_completed:
        moose_runs = [r for r in pending_runs if not _has_finished_moose_output(r)]
        reused_moose = len(pending_runs) - len(moose_runs)
        if reused_moose:
            print(f"  Reusing {reused_moose} completed MOOSE outputs; post-processing them only.")
    else:
        moose_runs = pending_runs

    arg_list = [
        (
            [f"tip_voltage={r['tip_voltage']}", f"pulse_end={r['pulse_end']}"]
            + [f"{k}={v}" for k, v in r["params"].items()]
            + [f"{k}={v}" for k, v in r.get("input_params", {}).items()]
            + [f"pfm_image_file={r['pfm_image_file']}"]
        )
        for r in moose_runs
    ]
    pp_arg_list = [
        (
            [f"--bias-voltage={r['tip_voltage']}"]
            + [f"--{k}={v}" for k, v in r["params"].items()]
            + [f"--tiled-path=phasefield/{r['key']}_surface_final"]
        )
        for r in pending_runs
    ]
    directory_list = [r["output_dir"] for r in pending_runs]

    if moose_runs:
        params = {
            "total_jobs": len(moose_runs),
            "base_input": manifest["base_input"],
            "arg_list": arg_list,
            "num_cores": manifest["num_cores"],
            "directory_list": [r["output_dir"] for r in moose_runs],
        }
        print(f"  Launching {len(moose_runs)} MOOSE jobs via MatEnsemble...")
        automoose(manifest["executable"], params)
    else:
        print("  All pending runs already have completed MOOSE outputs; skipping MOOSE launch.")

    for directory in directory_list:
        out_dir = Path(directory)
        for name in ("stdout", "stderr"):
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / f"moose_{name}")

    print("  Post-processing Exodus outputs...")
    postproc(
        ppscript=manifest["postprocess_script"],
        params={
            "total_jobs": len(runs),
            "base_input": manifest["base_input"],
            "arg_list": pp_arg_list,
            "num_cores": 1,
            "directory_list": directory_list,
        },
    )


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _load_surface(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    for key in ("disp_z", "uz", "polar_z"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    raise KeyError(f"No recognized surface field in {path}")


def _resample(src: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return resample_array(src, shape)


def _znorm(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation. Returns x unchanged if std≈0."""
    s = float(x.std())
    return (x - float(x.mean())) / (s if s > 1e-12 else 1.0)


def _normalized_mse(sim: np.ndarray, ref: np.ndarray) -> float:
    """Pattern-match loss: MSE of z-normalised fields.

    Removes scale and offset so the metric reflects domain structure similarity,
    not absolute amplitude. Range: 0 (perfect) → 2 (uncorrelated) → 4 (anti-correlated).
    Experimental PFM uz (~pm) and simulated uz (~nm) are on different absolute
    scales; raw MSE would be dominated by that offset rather than pattern quality.
    """
    return float(np.mean((_znorm(sim) - _znorm(ref)) ** 2))


def _center_crop_pair(
    sim: np.ndarray,
    ref: np.ndarray,
    crop_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    crop_fraction = max(float(crop_fraction), 0.0)
    if crop_fraction <= 0.0:
        return sim, ref
    ny, nx = sim.shape
    by = min(max(int(round(crop_fraction * ny)), 0), max((ny - 1) // 2, 0))
    bx = min(max(int(round(crop_fraction * nx)), 0), max((nx - 1) // 2, 0))
    if by == 0 and bx == 0:
        return sim, ref
    ys = slice(by, ny - by if by else ny)
    xs = slice(bx, nx - bx if bx else nx)
    return sim[ys, xs], ref[ys, xs]


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _read_named_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data is None or data.dtype.names is None:
        raise ValueError(f"CSV has no named columns: {path}")
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names}


def _energy_csv_path(output_dir: Path) -> Path | None:
    preferred = output_dir / "out_bto_wall_T298K_2.csv"
    if preferred.exists():
        return preferred
    matches = sorted(output_dir.glob("*.csv"))
    return matches[0] if matches else None


def _energy_summary(output_dir: Path, columns: list[str] | None = None) -> dict[str, Any] | None:
    csv_path = _energy_csv_path(output_dir)
    if csv_path is None:
        return None

    table = _read_named_csv(csv_path)
    time_values = table.get("time")
    if time_values is None or time_values.size == 0:
        return None

    energy_columns = columns or ["Fb", "Fw", "Fela", "Fc", "Fele"]
    present = [col for col in energy_columns if col in table]
    if present:
        free_energy = np.zeros_like(table[present[0]], dtype=np.float64)
        for col in present:
            free_energy = free_energy + table[col]
        total_name = "+".join(present)
    elif "Ftot" in table:
        free_energy = np.asarray(table["Ftot"], dtype=np.float64)
        total_name = "Ftot"
    else:
        return None

    order = np.argsort(time_values)
    time_values = time_values[order]
    free_energy = free_energy[order]
    moose_ftot = np.asarray(table["Ftot"], dtype=np.float64)[order] if "Ftot" in table else None

    summary: dict[str, Any] = {
        "csv_path": str(csv_path),
        "total_name": total_name,
        "columns": present if present else ["Ftot"],
        "n_steps": int(time_values.size),
        "time_initial": float(time_values[0]),
        "time_final": float(time_values[-1]),
        "free_energy_initial": float(free_energy[0]),
        "free_energy_final": float(free_energy[-1]),
        "free_energy_delta": float(free_energy[-1] - free_energy[0]),
        "free_energy_min": float(np.min(free_energy)),
        "free_energy_max": float(np.max(free_energy)),
    }
    if moose_ftot is not None:
        summary.update({
            "moose_Ftot_initial": float(moose_ftot[0]),
            "moose_Ftot_final": float(moose_ftot[-1]),
            "moose_Ftot_delta": float(moose_ftot[-1] - moose_ftot[0]),
            "note": "free_energy_* sums Fb+Fw+Fela+Fc+Fele when available; moose_Ftot is the MOOSE Ftot postprocessor.",
        })
    return summary


def _energy_reference_loss(sim_summary: dict[str, Any], target_final: Any) -> float | None:
    target = _finite_float(target_final)
    if target is None:
        return None
    sim_final = _finite_float(sim_summary.get("free_energy_final"))
    if sim_final is None:
        return None
    scale = max(abs(target), abs(sim_final), 1.0)
    return float(((sim_final - target) / scale) ** 2)


def compute_round_losses(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    loss_cfg = manifest.get("loss", {})
    surface_weight = float(loss_cfg.get("surface_uz_weight", 1.0))
    energy_weight = float(loss_cfg.get("free_energy_weight", 0.0))
    energy_columns = loss_cfg.get("free_energy_total_columns")
    energy_mode = str(loss_cfg.get("free_energy_mode", "diagnostic"))
    orientation = surface_uz_orientation(manifest)
    crop_fraction = float(loss_cfg.get("surface_uz_edge_crop_fraction", 0.0))
    objective_region = str(loss_cfg.get("surface_uz_objective_region", "full"))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for run in manifest["runs"]:
        grouped[run["candidate_id"]].append(run)

    observations: list[dict[str, Any]] = []
    for cand_id, runs in grouped.items():
        losses: list[float] = []
        energy_losses: list[float] = []
        condition_metrics: list[dict[str, Any]] = []
        for run in runs:
            sim_path = Path(run["output_dir"]) / "fields_final_timestep.npz"
            ref_path = Path(run["experimental_uz_path"])
            condition_metric: dict[str, Any] = {
                "experiment_key": run.get("experiment_key"),
                "tip_voltage": run.get("tip_voltage"),
                "pulse_end": run.get("pulse_end"),
            }
            if not sim_path.exists():
                print(f"    WARN: missing sim output {sim_path}")
                condition_metrics.append(condition_metric)
                continue
            try:
                sim = _load_surface(sim_path)
                sim = orient_sim_for_experiment(sim, orientation)
                ref = _load_surface(ref_path)
                if sim.shape != ref.shape:
                    ref = _resample(ref, sim.shape)
                full_surface_loss = _normalized_mse(sim, ref)
                surface_loss = full_surface_loss
                if crop_fraction > 0.0:
                    sim_crop, ref_crop = _center_crop_pair(sim, ref, crop_fraction)
                    center_surface_loss = _normalized_mse(sim_crop, ref_crop)
                    condition_metric["surface_uz_nmse_full"] = full_surface_loss
                    condition_metric["surface_uz_nmse_center_crop"] = center_surface_loss
                    condition_metric["surface_uz_edge_crop_fraction"] = crop_fraction
                    if objective_region in {"center", "center_crop", "masked", "center_80pct"}:
                        surface_loss = center_surface_loss
                losses.append(surface_loss)
                condition_metric["surface_uz_nmse"] = surface_loss
                condition_metric["surface_uz_orientation"] = orientation
                condition_metric["surface_uz_objective_region"] = (
                    "center_crop" if crop_fraction > 0.0 and surface_loss != full_surface_loss else "full"
                )
            except Exception as exc:
                print(f"    WARN: loss computation failed for {run['key']}: {exc}")

            try:
                summary = _energy_summary(Path(run["output_dir"]), energy_columns)
                if summary is not None:
                    condition_metric["free_energy"] = summary
                    if energy_mode == "final_reference":
                        target_final = run.get("free_energy_final_reference")
                        energy_loss = _energy_reference_loss(summary, target_final)
                        if energy_loss is not None:
                            energy_losses.append(energy_loss)
                            condition_metric["free_energy_loss"] = energy_loss
            except Exception as exc:
                print(f"    WARN: energy summary failed for {run['key']}: {exc}")

            condition_metrics.append(condition_metric)

        first = runs[0]
        params = {k: float(v) for k, v in first["params"].items()}
        if not losses:
            # All MOOSE runs for this candidate failed — record a large penalty
            # so the GP avoids this region and it won't be re-selected.
            print(f"    WARN: all {len(runs)} runs failed for {cand_id} — recording penalty.")
            observations.append({
                "candidate_id": cand_id,
                "parameters": params,
                "objective": 1e6,
                "objective_name": "surface_uz_nmse",
                "maximize": False,
                "n_conditions": 0,
                "losses_per_condition": [],
                "condition_metrics": condition_metrics,
                "converged": False,
            })
            continue
        surface_objective = float(np.mean(losses))
        objective = surface_weight * surface_objective
        objective_name = "surface_uz_nmse"
        energy_objective = None
        if energy_weight > 0.0:
            if energy_losses:
                energy_objective = float(np.mean(energy_losses))
                objective += energy_weight * energy_objective
                objective_name = "surface_uz_nmse_plus_free_energy"
            else:
                print(
                    "    WARN: free_energy_weight > 0 but no free-energy reference "
                    f"losses were available for {cand_id}; objective uses surface uz only."
                )

        obs = {
            "candidate_id": cand_id,
            "parameters": params,
            "objective": float(objective),
            "objective_name": objective_name,
            "surface_uz_objective": surface_objective,
            "maximize": False,
            "n_conditions": len(losses),
            "losses_per_condition": losses,
            "condition_metrics": condition_metrics,
            "converged": True,
        }
        if energy_objective is not None:
            obs["free_energy_objective"] = energy_objective
        observations.append(obs)
    return observations


# ---------------------------------------------------------------------------
# BO state update
# ---------------------------------------------------------------------------

def update_bo_state(
    bo_state: dict[str, Any],
    bo_state_path: Path,
    observations: list[dict[str, Any]],
    round_idx: int,
) -> None:
    for obs in observations:
        bo_state["observations"].append({
            "round": round_idx,
            "parameters": obs["parameters"],
            "objective": obs["objective"],
            "n_conditions": obs["n_conditions"],
        })
    _save_bo_state(bo_state, bo_state_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_params(params: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.5g}" for k, v in params.items())


def _print_round_summary(round_idx: int, candidates: list[dict], observations: list[dict]) -> None:
    print(f"\n  Round {round_idx} results:")
    for obs in sorted(observations, key=lambda o: o["objective"]):
        tag = "" if obs.get("converged", True) else "  [FAILED]"
        print(f"    {obs['candidate_id']}  {_format_params(obs['parameters'])}"
              f"  MSE={obs['objective']:.6e}  ({obs['n_conditions']} cond){tag}")


def _print_final_summary(bo_state: dict[str, Any]) -> None:
    obs = bo_state["observations"]
    if not obs:
        print("No observations recorded.")
        return
    converged = [o for o in obs if o.get("converged", True) and o["n_conditions"] > 0]
    best = min(converged or obs, key=lambda o: o["objective"])
    print("\n" + "=" * 70)
    print("BO COMPLETE — BEST PARAMETERS FOUND")
    print("=" * 70)
    for k, v in best["parameters"].items():
        print(f"  {k} = {v:.6f}")
    print(f"  surface_uz_nMSE = {best['objective']:.6e}  "
          f"(round {best.get('round', '?')}, {best['n_conditions']} conditions)")
    print(f"\n  Total observations: {len(obs)} across "
          f"{1 + max(o.get('round', 0) for o in obs)} round(s)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous simulation-only BO loop on NERSC.")
    parser.add_argument("--config",
                        default=os.environ.get("SIM_BO_CONFIG", "sim_bo_config.json"),
                        help="Config file (default: $SIM_BO_CONFIG or sim_bo_config.json).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config_path = config_path.resolve()
    campaign_dir = config_path.parent

    print("=" * 70)
    print("SIMULATION-ONLY BAYESIAN OPTIMIZATION LOOP")
    print("=" * 70)
    print(f"  Config:           {config_path}")
    print(f"  Campaign dir:     {campaign_dir}")

    cfg = _load_config(config_path)
    n_rounds = cfg["n_rounds"]
    n_train = len(cfg["train_refs"])
    n_holdout = len(cfg["holdout_refs"])
    n_nodes = int(os.environ.get("SLURM_NNODES", "4"))
    n_workers = max(1, n_nodes - 1)  # one MatEnsemble root, remaining nodes as workers
    runs_per_round = cfg["batch_size"] * n_train
    passes_per_round = -(-runs_per_round // n_workers)   # ceil
    print(f"  Rounds:           {n_rounds}")
    print(f"  Batch size:       {cfg['batch_size']} candidates/round")
    print(f"  Train conditions: {n_train}  |  Holdout: {n_holdout}")
    print(f"  MOOSE runs/round: {runs_per_round}  ({passes_per_round} passes × {n_workers} workers)")
    print(f"  Executable:       {cfg['executable']}")
    print(f"  Train keys:  {[r['key'] for r in cfg['train_refs']]}")
    print(f"  Holdout keys:{[r['key'] for r in cfg['holdout_refs']]}")

    n_params = len(_param_keys(cfg))
    n_samples = cfg.get("num_candidate_samples", 8192)
    print(f"  Parameters ({n_params}): {_param_keys(cfg)}")
    print(f"  Candidate samples/round: {n_samples}  (Sobol quasi-random)")
    rng = np.random.default_rng(cfg["random_seed"])

    bo_state_path = campaign_dir / "bo_state.json"
    try:
        bo_state = _load_bo_state(bo_state_path)
        # Resume support: find the highest round already recorded in bo_state
        # and skip ahead so we don't duplicate completed work.
        completed_rounds = {o["round"] for o in bo_state["observations"] if isinstance(o.get("round"), int)}
        start_round = max(completed_rounds) + 1 if completed_rounds else 0
    except:
        start_round=0

    if start_round > 0:
        print(f"\n  Resuming from round {start_round + 1} "
              f"({start_round} rounds already in bo_state.json)")
        # Fast-forward the RNG to stay consistent with the original sequence
        for _ in range(start_round):
            _sample_candidates(cfg, n_samples=cfg.get("num_candidate_samples", 8192), rng=rng)

    t_start = time.monotonic()

    # ── BO rounds (training conditions only) ──────────────────────────────
    for round_idx in range(start_round, n_rounds):
        print(f"\n{'─' * 70}")
        print(f"ROUND {round_idx + 1} / {n_rounds}  "
              f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC]")
        print(f"{'─' * 70}")

        candidates = select_candidates(bo_state, cfg, rng)
        print(f"  Candidates selected ({len(candidates)}):")
        for i, c in enumerate(candidates):
            print(f"    [{i}] {_format_params(c)}")

        # Build manifest using TRAIN refs only
        manifest = build_round_manifest(round_idx, candidates, cfg, campaign_dir,
                                        ref_key="train_refs")

        t_round = time.monotonic()
        try:
            run_round(manifest)
        except Exception as exc:
            print(f"  WARN: run_round raised {type(exc).__name__}: {exc}")
            print("  Continuing with whatever outputs were written...")
        print(f"  MOOSE + postproc finished in {(time.monotonic() - t_round) / 60:.1f} min")

        observations = compute_round_losses(manifest)
        if not observations:
            print("  WARN: no valid observations this round — check MOOSE outputs.")
            continue

        update_bo_state(bo_state, bo_state_path, observations, round_idx)
        _print_round_summary(round_idx, candidates, observations)

        round_obs_path = campaign_dir / f"round_{round_idx:03d}" / "observations.json"
        round_obs_path.parent.mkdir(parents=True, exist_ok=True)
        round_obs_path.write_text(json.dumps(observations, indent=2), encoding="utf-8")

    # ── Holdout evaluation ────────────────────────────────────────────────
    if bo_state["observations"] and cfg.get("holdout_refs"):
        print(f"\n{'─' * 70}")
        print("HOLDOUT EVALUATION — top-3 candidates vs unseen conditions")
        print(f"{'─' * 70}")

        # Pick the top-3 unique parameter sets by training MSE
        seen: set[tuple] = set()
        top_candidates: list[dict[str, float]] = []
        for obs in sorted(
            (o for o in bo_state["observations"] if o.get("converged", True) and o["n_conditions"] > 0),
            key=lambda o: o["objective"],
        ):
            key = tuple(round(obs["parameters"][k], 6) for k in sorted(obs["parameters"]))
            if key not in seen:
                seen.add(key)
                top_candidates.append(obs["parameters"])
            if len(top_candidates) == 3:
                break

        holdout_manifest = build_round_manifest(
            "holdout", top_candidates, cfg, campaign_dir, ref_key="holdout_refs"
        )
        t_holdout = time.monotonic()
        try:
            run_round(holdout_manifest)
        except Exception as exc:
            print(f"  WARN: holdout run_round raised {type(exc).__name__}: {exc}")
        print(f"  Holdout MOOSE finished in {(time.monotonic() - t_holdout) / 60:.1f} min")

        holdout_obs = compute_round_losses(holdout_manifest)
        print(f"\n  Holdout results ({len(cfg['holdout_refs'])} unseen conditions):")
        for obs in sorted(holdout_obs, key=lambda o: o["objective"]):
            print(f"    {_format_params(obs['parameters'])}"
                  f"  holdout_MSE={obs['objective']:.6e}  ({obs['n_conditions']} cond)")

        holdout_path = campaign_dir / "holdout_results.json"
        holdout_path.write_text(json.dumps(holdout_obs, indent=2), encoding="utf-8")
        print(f"\n  Holdout results written to: {holdout_path}")

    print(f"\n  Total wall time: {(time.monotonic() - t_start) / 3600:.2f} h")
    _print_final_summary(bo_state)


if __name__ == "__main__":
    main()
