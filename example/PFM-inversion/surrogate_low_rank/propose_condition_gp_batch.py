#!/usr/bin/env python3
"""Propose and optionally run new candidates using the condition-aware GP.

Acquisition score for a candidate theta is

    A(theta) = mean_mse(theta) - kappa * sqrt(mean_field_variance(theta))

so lower scores are selected.  With kappa=0 this is pure surrogate inverse-fit;
larger kappa explores uncertain regions in the physical parameter box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gpytorch
import numpy as np
import torch

import train_uz_gp_surrogate as gp_train
import train_uz_svd_surrogate as svd
from inverse_fit_g_from_pfm import localize_ref_path, resample, torch_dtype, select_device
from run_bo_loop import build_round_manifest, compute_round_losses, run_round
from surface_orientation import orient_experiment_for_raw_sim, surface_uz_orientation


PARAM_KEYS = ("g11", "g12", "g44")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unique_refs(config: dict, which: str) -> list[dict]:
    refs = config["train_refs"] if which == "train" else config["train_refs"] + config.get("holdout_refs", [])
    seen = set()
    out = []
    for ref in refs:
        if ref["key"] in seen:
            continue
        seen.add(ref["key"])
        out.append(ref)
    return out


def bounds(config: dict):
    lo = np.asarray([config["parameter_space"][k]["lower"] for k in PARAM_KEYS], dtype=np.float64)
    hi = np.asarray([config["parameter_space"][k]["upper"] for k in PARAM_KEYS], dtype=np.float64)
    return lo, hi


def load_targets(config: dict, refs: list[dict], campaign_dir: Path, shape, normalization):
    fields = []
    conditions = []
    orientation = surface_uz_orientation(config)
    for ref in refs:
        path = localize_ref_path(campaign_dir, ref["npz_path"])
        arr = svd.load_surface(path, "auto")
        arr = orient_experiment_for_raw_sim(arr, tuple(shape), orientation)
        arr = svd.normalize_field(arr, normalization)
        fields.append(arr.reshape(-1))
        conditions.append([float(ref["tip_voltage"]), float(ref["pulse_end"])])
    return np.vstack(fields).astype(np.float64), np.asarray(conditions, dtype=np.float64)


def load_gp(gp_dir: Path, device: torch.device, dtype: torch.dtype):
    state = torch.load(gp_dir / "condition_field_gp_state.pt", map_location=device)
    prep = np.load(gp_dir / "condition_field_preprocess.npz", allow_pickle=False)
    inducing = state["inducing_points"].to(device=device, dtype=dtype)
    model = gp_train.IndependentMultitaskSVDDKL(
        inducing,
        num_tasks=int(state["num_tasks"]),
    ).to(device=device, dtype=dtype)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=int(state["num_tasks"]),
    ).to(device=device, dtype=dtype)
    model.load_state_dict(state["model_state_dict"])
    likelihood.load_state_dict(state["likelihood_state_dict"])
    model.eval()
    likelihood.eval()
    return model, likelihood, prep


def random_candidates(n: int, lo: np.ndarray, hi: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    try:
        from scipy.stats.qmc import Sobol
        unit = Sobol(d=3, scramble=True, seed=seed).random(n)
    except Exception:
        unit = rng.uniform(0.0, 1.0, (n, 3))
    return unit * (hi - lo) + lo


def load_evaluated_units(campaign_dir: Path, lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    """Return normalized coordinates for previously observed candidates."""
    units: list[np.ndarray] = []
    obs_paths = sorted(campaign_dir.glob("round_*/observations*.json"))
    for extra in ("anchor_results.json",):
        path = campaign_dir / extra
        if path.exists():
            obs_paths.append(path)

    for path in obs_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            params = row.get("parameters") if isinstance(row, dict) else None
            if not isinstance(params, dict) or not all(k in params for k in PARAM_KEYS):
                continue
            g = np.asarray([float(params[k]) for k in PARAM_KEYS], dtype=np.float64)
            if np.any(g < lo) or np.any(g > hi):
                continue
            units.append((g - lo) / (hi - lo))
    return units


def score_candidates(args):
    campaign_dir = Path(args.campaign_dir).resolve()
    config = load_json(Path(args.config).resolve() if Path(args.config).is_absolute() else campaign_dir / args.config)
    gp_dir = Path(args.gp_dir).resolve() if Path(args.gp_dir).is_absolute() else campaign_dir / args.gp_dir
    device = select_device(args.device)
    dtype = torch_dtype(args.dtype)

    model, likelihood, prep = load_gp(gp_dir, device, dtype)
    normalization = str(prep["normalization"][0]) if "normalization" in prep.files else "znorm"
    refs = unique_refs(config, args.refs)
    target_np, conditions_np = load_targets(
        config,
        refs,
        campaign_dir,
        shape=prep["image_shape"],
        normalization=normalization,
    )

    lo, hi = bounds(config)
    candidates = random_candidates(args.num_candidate_samples, lo, hi, args.random_seed)
    anchor_candidates = [a["parameters"] for a in config.get("anchor_candidates", [])]
    if anchor_candidates:
        anchor_array = np.asarray([[a[k] for k in PARAM_KEYS] for a in anchor_candidates], dtype=np.float64)
        candidates = np.vstack([anchor_array, candidates])

    input_mean = torch.as_tensor(prep["input_mean"], dtype=dtype, device=device)
    input_scale = torch.as_tensor(prep["input_scale"], dtype=dtype, device=device)
    coeff_mean = torch.as_tensor(prep["coeff_mean"], dtype=dtype, device=device)
    coeff_scale = torch.as_tensor(prep["coeff_scale"], dtype=dtype, device=device)
    field_mean = torch.as_tensor(prep["field_mean"], dtype=dtype, device=device)
    basis = torch.as_tensor(prep["basis"], dtype=dtype, device=device)
    basis_norm = torch.sum(basis * basis, dim=1)
    target = torch.as_tensor(target_np, dtype=dtype, device=device)
    conditions = torch.as_tensor(conditions_np, dtype=dtype, device=device)

    rows = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, candidates.shape[0], args.eval_batch_size):
            batch = candidates[start:start + args.eval_batch_size]
            theta = torch.as_tensor(batch, dtype=dtype, device=device)
            raw_x_parts = []
            for row in theta:
                raw_x_parts.append(torch.cat([row.reshape(1, 3).expand(conditions.size(0), 3), conditions], dim=1))
            raw_x = torch.cat(raw_x_parts, dim=0)
            x = (raw_x - input_mean) / input_scale
            pred_dist = likelihood(model(x))
            coeff = pred_dist.mean * coeff_scale + coeff_mean
            pred = field_mean + coeff @ basis
            residual = pred - target.repeat(theta.size(0), 1)
            mse_by_condition = torch.mean(residual * residual, dim=1).reshape(theta.size(0), conditions.size(0))
            mse = torch.mean(mse_by_condition, dim=1)
            coeff_var = pred_dist.variance * (coeff_scale.reshape(1, -1) ** 2)
            field_var = ((coeff_var @ basis_norm) / basis.size(1)).reshape(theta.size(0), conditions.size(0))
            mean_var = torch.mean(field_var, dim=1)
            score = mse - float(args.kappa) * torch.sqrt(torch.clamp(mean_var, min=0.0))
            for i in range(theta.size(0)):
                g = batch[i]
                rows.append({
                    "parameters": {k: float(v) for k, v in zip(PARAM_KEYS, g)},
                    "predicted_mse": float(mse[i].detach().cpu()),
                    "mean_field_variance": float(mean_var[i].detach().cpu()),
                    "acquisition": float(score[i].detach().cpu()),
                })

    rows.sort(key=lambda row: row["acquisition"])
    excluded_unit = load_evaluated_units(campaign_dir, lo, hi) if args.exclude_evaluated else []
    selected = []
    selected_unit = []
    for row in rows:
        g = np.asarray([row["parameters"][k] for k in PARAM_KEYS], dtype=np.float64)
        unit = (g - lo) / (hi - lo)
        if excluded_unit:
            distances = [float(np.linalg.norm(unit - prev)) for prev in excluded_unit]
            if min(distances) < args.min_normalized_distance:
                continue
        if selected_unit:
            distances = [float(np.linalg.norm(unit - prev)) for prev in selected_unit]
            if min(distances) < args.min_normalized_distance:
                continue
        selected.append(row)
        selected_unit.append(unit)
        if len(selected) >= args.batch_size:
            break

    out_dir = campaign_dir / "condition_gp_active"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(campaign_dir / args.config),
        "gp_dir": str(gp_dir),
        "refs": args.refs,
        "kappa": float(args.kappa),
        "num_candidate_samples": int(args.num_candidate_samples),
        "batch_size": int(args.batch_size),
        "exclude_evaluated": bool(args.exclude_evaluated),
        "excluded_evaluated_count": int(len(excluded_unit)),
        "selected": selected,
        "top20": rows[:20],
    }
    proposal_path = out_dir / f"{args.round_name}_proposal.json"
    proposal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest = None
    observations = None
    if args.build_manifest or args.run:
        manifest = build_round_manifest(
            args.round_name,
            [row["parameters"] for row in selected],
            config,
            campaign_dir,
            ref_key="train_refs",
        )
        manifest_path = campaign_dir / f"round_{args.round_name}" / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        payload["manifest"] = str(manifest_path)
        proposal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.run:
        run_round(manifest)
        observations = compute_round_losses(manifest)
        obs_path = campaign_dir / f"round_{args.round_name}" / "observations.json"
        obs_path.write_text(json.dumps(observations, indent=2), encoding="utf-8")
        payload["observations"] = str(obs_path)
        proposal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote proposal: {proposal_path}")
    print("Selected candidates:")
    for i, row in enumerate(selected):
        p = row["parameters"]
        print(
            f"  [{i}] g11={p['g11']:.6g} g12={p['g12']:.6g} g44={p['g44']:.6g} "
            f"mse={row['predicted_mse']:.6e} var={row['mean_field_variance']:.6e} "
            f"acq={row['acquisition']:.6e}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--config", default="sim_bo_config_interactive.json")
    parser.add_argument("--gp-dir", default="gp_surrogate_combined")
    parser.add_argument("--refs", choices=("train", "all"), default="train")
    parser.add_argument("--num-candidate-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--min-normalized-distance", type=float, default=0.08)
    parser.add_argument("--exclude-evaluated", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=20260604)
    parser.add_argument("--round-name", default="gp_active_000")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    score_candidates(args)


if __name__ == "__main__":
    main()
