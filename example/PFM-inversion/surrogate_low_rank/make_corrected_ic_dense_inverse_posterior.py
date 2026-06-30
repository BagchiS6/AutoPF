#!/usr/bin/env python3
"""Dense corrected-IC inverse posterior from a corrected-anchor POD surrogate.

This is the corrected-IC analogue of ``inverse_posterior_transpose``.  The old
posterior was produced from a condition-aware GP/POD model trained on a much
larger campaign.  Here we train a small POD + RBF-kernel surrogate only on the
corrected-IC g_ij anchor round, then evaluate a dense Sobol grid to produce the
same style of ``candidate_scores.csv``, ``posterior_summary.json``, and
``loss_profiles.png``.

Because the corrected g_ij-only training set currently contains only six MOOSE
candidates, this should be read as an interpolation diagnostic, not as a final
well-identified posterior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simbo")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from surface_orientation import orient_sim_for_experiment, resample_array


ROOT = Path(__file__).resolve().parent
PARAM_KEYS = ("g11", "g12", "g44")
BULK_BTO = {"g11": 0.5, "g12": -0.02, "g44": 0.02}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def znorm(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    std = float(np.nanstd(array))
    return (array - float(np.nanmean(array))) / (std if std > 1.0e-12 else 1.0)


def crop_slices(shape: tuple[int, int], crop_fraction: float) -> tuple[slice, slice]:
    ny, nx = shape
    by = min(max(int(round(float(crop_fraction) * ny)), 0), max((ny - 1) // 2, 0))
    bx = min(max(int(round(float(crop_fraction) * nx)), 0), max((nx - 1) // 2, 0))
    return slice(by, ny - by if by else ny), slice(bx, nx - bx if bx else nx)


def load_surface_pair(run: dict[str, Any], orientation: str) -> tuple[np.ndarray, np.ndarray]:
    fields = np.load(Path(run["output_dir"]) / "fields_final_timestep.npz", allow_pickle=False)
    sim_key = "disp_z" if "disp_z" in fields else "uz"
    sim = orient_sim_for_experiment(np.asarray(fields[sim_key], dtype=np.float64), orientation)
    exp = np.asarray(np.load(run["experimental_uz_path"], allow_pickle=False)["uz"], dtype=np.float64)
    sim = resample_array(sim, exp.shape)
    return znorm(sim), znorm(exp)


def corrected_anchor_dataset(campaign_dir: Path, round_name: str) -> dict[str, Any]:
    round_dir = campaign_dir / f"round_{round_name}"
    manifest = load_json(round_dir / "manifest.json")
    observations = load_json(round_dir / "observations.json")
    if isinstance(observations, dict):
        observations = observations.get("observations", [])
    observation_by_candidate = {row["candidate_id"]: row for row in observations}
    orientation = str(manifest.get("loss", {}).get("surface_uz_orientation", "identity"))
    crop_fraction = float(manifest.get("loss", {}).get("surface_uz_edge_crop_fraction", 0.0))

    runs_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for run in manifest.get("runs", []):
        runs_by_candidate.setdefault(run["candidate_id"], []).append(run)

    condition_keys = [
        run["experiment_key"]
        for run in sorted(next(iter(runs_by_candidate.values())), key=lambda item: (float(item["tip_voltage"]), float(item["pulse_end"])))
    ]
    X_rows = []
    Y_rows = []
    candidate_ids = []
    experimental_fields: list[np.ndarray] | None = None
    shape: tuple[int, int] | None = None

    for candidate_id in sorted(runs_by_candidate):
        runs = sorted(runs_by_candidate[candidate_id], key=lambda item: (float(item["tip_voltage"]), float(item["pulse_end"])))
        if [run["experiment_key"] for run in runs] != condition_keys:
            continue
        params = runs[0]["params"]
        X_rows.append([float(params[key]) for key in PARAM_KEYS])
        candidate_ids.append(candidate_id)
        sim_fields = []
        exp_fields = []
        for run in runs:
            sim, exp = load_surface_pair(run, orientation)
            shape = sim.shape
            sim_fields.append(sim.reshape(-1))
            exp_fields.append(exp.reshape(-1))
        if experimental_fields is None:
            experimental_fields = exp_fields
        Y_rows.append(np.concatenate(sim_fields))

    if not X_rows or experimental_fields is None or shape is None:
        raise RuntimeError(f"No corrected anchor records found in {round_dir}")

    exp_concat = np.concatenate(experimental_fields)
    crop_y, crop_x = crop_slices(shape, crop_fraction)
    mask_one = np.zeros(shape, dtype=bool)
    mask_one[crop_y, crop_x] = True
    mask = np.concatenate([mask_one.reshape(-1) for _ in condition_keys])
    return {
        "X": np.asarray(X_rows, dtype=np.float64),
        "Y": np.vstack(Y_rows).astype(np.float64),
        "candidate_ids": candidate_ids,
        "experimental": exp_concat.astype(np.float64),
        "objective_mask": mask,
        "condition_keys": condition_keys,
        "image_shape": list(shape),
        "crop_fraction": crop_fraction,
        "orientation": orientation,
        "observations": observation_by_candidate,
    }


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale[scale <= 1.0e-12] = 1.0
    return (X - mean) / scale, mean, scale


def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A2 = np.sum(A * A, axis=1).reshape(-1, 1)
    B2 = np.sum(B * B, axis=1).reshape(1, -1)
    return np.maximum(A2 + B2 - 2.0 * np.dot(A, B.T), 0.0)


def median_pairwise_distance(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 1.0
    d2 = squared_distances(X, X)
    tri = np.triu_indices(X.shape[0], k=1)
    vals = np.sqrt(d2[tri])
    vals = vals[vals > 1.0e-12]
    return float(np.median(vals)) if vals.size else 1.0


def rbf(A: np.ndarray, B: np.ndarray, length_scale: float) -> np.ndarray:
    return np.exp(-0.5 * squared_distances(A, B) / (float(length_scale) ** 2))


def compute_pod(Y: np.ndarray, variance: float, max_components: int) -> dict[str, np.ndarray | float]:
    mean = np.mean(Y, axis=0)
    centered = Y - mean
    gram = np.dot(centered, centered.T)
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    svals = np.sqrt(evals)
    keep = svals > 1.0e-12
    svals = svals[keep]
    evecs = evecs[:, keep]
    if svals.size == 0:
        raise RuntimeError("Corrected-IC POD has no nonzero modes")
    basis_all = (centered.T @ evecs / svals).T
    frac = (svals**2) / float(np.sum(svals**2))
    cumulative = np.cumsum(frac)
    k = min(int(np.searchsorted(cumulative, variance) + 1), int(max_components), svals.size)
    k = max(k, 1)
    basis = basis_all[:k]
    coeff = (Y - mean) @ basis.T
    return {
        "mean": mean,
        "basis": basis,
        "singular_values": svals[:k],
        "explained_variance": frac[:k],
        "retained_energy": float(cumulative[k - 1]),
        "coefficients": coeff,
    }


def fit_surrogate(dataset: dict[str, Any], alpha: float, variance: float, max_components: int) -> dict[str, Any]:
    X = dataset["X"]
    X_scaled, input_mean, input_scale = standardize(X)
    pod = compute_pod(dataset["Y"], variance=variance, max_components=max_components)
    coeff = np.asarray(pod["coefficients"], dtype=np.float64)
    coeff_mean = np.mean(coeff, axis=0)
    coeff_scale = np.std(coeff, axis=0)
    coeff_scale[coeff_scale <= 1.0e-12] = 1.0
    coeff_scaled = (coeff - coeff_mean) / coeff_scale
    length_scale = median_pairwise_distance(X_scaled)
    kernel = rbf(X_scaled, X_scaled, length_scale)
    kernel += np.eye(kernel.shape[0]) * float(alpha)
    dual = np.linalg.solve(kernel, coeff_scaled)
    return {
        "X_train": X,
        "X_train_scaled": X_scaled,
        "input_mean": input_mean,
        "input_scale": input_scale,
        "length_scale": length_scale,
        "alpha": float(alpha),
        "dual": dual,
        "coeff_mean": coeff_mean,
        "coeff_scale": coeff_scale,
        "field_mean": pod["mean"],
        "basis": pod["basis"],
        "singular_values": pod["singular_values"],
        "explained_variance": pod["explained_variance"],
        "retained_energy": pod["retained_energy"],
    }


def predict_fields(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X_scaled = (X - model["input_mean"]) / model["input_scale"]
    coeff_scaled = rbf(X_scaled, model["X_train_scaled"], model["length_scale"]) @ model["dual"]
    coeff = coeff_scaled * model["coeff_scale"] + model["coeff_mean"]
    return model["field_mean"] + coeff @ model["basis"]


def bounds_from_config(campaign_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    config = load_json(campaign_dir / "sim_bo_config.json")
    lo = np.asarray([config["parameter_space"][key]["lower"] for key in PARAM_KEYS], dtype=np.float64)
    hi = np.asarray([config["parameter_space"][key]["upper"] for key in PARAM_KEYS], dtype=np.float64)
    return lo, hi


def sample_grid(n: int, lo: np.ndarray, hi: np.ndarray, seed: int) -> np.ndarray:
    try:
        from scipy.stats.qmc import Sobol

        exponent = int(math.ceil(math.log(max(n, 2), 2)))
        unit = Sobol(d=3, scramble=True, seed=seed).random_base2(exponent)[:n]
    except Exception:
        unit = np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, 3))
    return unit * (hi - lo) + lo


def score_candidates(
    model: dict[str, Any],
    dataset: dict[str, Any],
    candidates: np.ndarray,
    batch_size: int,
) -> list[dict[str, float]]:
    target = np.asarray(dataset["experimental"], dtype=np.float64)
    mask = np.asarray(dataset["objective_mask"], dtype=bool)
    denom = float(np.mean(target[mask] ** 2))
    if denom <= 1.0e-12:
        denom = 1.0
    rows: list[dict[str, float]] = []
    for start in range(0, candidates.shape[0], batch_size):
        batch = candidates[start : start + batch_size]
        pred = predict_fields(model, batch)
        residual = pred[:, mask] - target[mask].reshape(1, -1)
        mse = np.mean(residual * residual, axis=1) / denom
        for values, loss in zip(batch, mse):
            rows.append(
                {
                    "g11": float(values[0]),
                    "g12": float(values[1]),
                    "g44": float(values[2]),
                    "mse": float(loss),
                    "loss": float(loss),
                }
            )
    rows.sort(key=lambda row: row["loss"])
    return rows


def profiles(rows: list[dict[str, float]], lo: np.ndarray, hi: np.ndarray, bins: int) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    for dim, key in enumerate(PARAM_KEYS):
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        edges = np.linspace(lo[dim], hi[dim], bins + 1)
        prof = []
        for idx in range(bins):
            if idx == bins - 1:
                mask = (vals >= edges[idx]) & (vals <= edges[idx + 1])
            else:
                mask = (vals >= edges[idx]) & (vals < edges[idx + 1])
            if not np.any(mask):
                continue
            prof.append(
                {
                    "center": float(0.5 * (edges[idx] + edges[idx + 1])),
                    "count": int(np.sum(mask)),
                    "min_mse": float(np.min(losses[mask])),
                    "mean_mse": float(np.mean(losses[mask])),
                }
            )
        out[key] = prof
    return out


def pseudo_posterior(rows: list[dict[str, float]], temperature: float) -> dict[str, dict[str, float]]:
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    weights = np.exp(-(losses - float(np.min(losses))) / max(float(temperature), 1.0e-12))
    weights = weights / float(np.sum(weights))
    stats = {}
    for key in PARAM_KEYS:
        vals = np.asarray([row[key] for row in rows], dtype=np.float64)
        order = np.argsort(vals)
        cdf = np.cumsum(weights[order])
        mean = float(np.sum(vals * weights))
        var = float(np.sum(((vals - mean) ** 2) * weights))
        stats[key] = {
            "mean": mean,
            "std": float(np.sqrt(max(var, 0.0))),
            "q05": float(vals[order][np.searchsorted(cdf, 0.05)]),
            "q50": float(vals[order][np.searchsorted(cdf, 0.50)]),
            "q95": float(vals[order][np.searchsorted(cdf, 0.95)]),
        }
    return stats


def write_scores(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["g11", "g12", "g44", "mse", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_profiles(summary: dict[str, Any], observed: dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    for ax, key in zip(axes, PARAM_KEYS):
        prof = summary["profiles"][key]
        x = np.asarray([row["center"] for row in prof], dtype=float)
        y_min = np.asarray([row["min_mse"] for row in prof], dtype=float)
        y_mean = np.asarray([row["mean_mse"] for row in prof], dtype=float)
        ax.plot(x, y_min, color="#4477aa", lw=2.0, label="bin min")
        ax.plot(x, y_mean, color="#999999", lw=1.5, label="bin mean")
        ax.axvline(summary["best"][key], color="#cc6677", lw=1.6, ls="--", label="dense surrogate best")
        ax.axvline(observed["best_observed"][key], color="#2ca02c", lw=1.3, ls=(0, (4, 2)), label="best MOOSE anchor")
        ax.axvline(BULK_BTO[key], color="#5b5f97", lw=1.2, ls=(0, (1.4, 2.0)), label="bulk BTO")
        ax.set_xlabel(key)
        ax.set_ylabel("surface uz nMSE")
        ax.set_title(f"corrected dense profile over {key}")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.4)
    fig.suptitle(
        "Corrected-IC dense inverse posterior from POD/RBF surrogate",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(out_dir / "loss_profiles.png", dpi=220)
    fig.savefig(out_dir / "loss_profiles.pdf")
    fig.savefig(ROOT / "corrected_ic_reanalysis" / "figures" / "generated" / "fig_corrected_ic_dense_loss_profiles.png", dpi=300)
    fig.savefig(ROOT / "corrected_ic_reanalysis" / "figures" / "generated" / "fig_corrected_ic_dense_loss_profiles.pdf")
    plt.close(fig)


def plot_pair_projections(rows: list[dict[str, float]], summary: dict[str, Any], out_dir: Path) -> None:
    losses = np.asarray([row["loss"] for row in rows], dtype=float)
    order = np.argsort(losses)
    keep = np.unique(np.r_[order[:6000], order[:: max(len(order) // 6000, 1)]])
    pairs = [("g11", "g12"), ("g11", "g44"), ("g12", "g44")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for ax, (xkey, ykey) in zip(axes, pairs):
        x = np.asarray([rows[i][xkey] for i in keep], dtype=float)
        y = np.asarray([rows[i][ykey] for i in keep], dtype=float)
        c = np.asarray([rows[i]["loss"] for i in keep], dtype=float)
        sc = ax.scatter(x, y, c=c, s=5, cmap="viridis_r", alpha=0.8, rasterized=True)
        ax.scatter(summary["best"][xkey], summary["best"][ykey], marker="*", s=90, color="#cc6677", edgecolor="black", linewidth=0.5)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=axes, label="loss", shrink=0.9)
    fig.savefig(out_dir / "loss_pair_projections.png", dpi=220)
    fig.savefig(out_dir / "loss_pair_projections.pdf")
    plt.close(fig)


def plot_marginals(rows: list[dict[str, float]], summary: dict[str, Any], out_dir: Path, temperature: float) -> None:
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    weights = np.exp(-(losses - float(np.min(losses))) / max(float(temperature), 1.0e-12))
    weights = weights / float(np.sum(weights))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, key in zip(axes, PARAM_KEYS):
        vals = np.asarray([row[key] for row in rows], dtype=float)
        ax.hist(vals, bins=45, weights=weights, color="#88ccee", edgecolor="none")
        ax.axvline(summary["best"][key], color="#cc6677", lw=1.5, ls="--", label="best")
        stats = summary["pseudo_posterior"][key]
        ax.axvspan(stats["q05"], stats["q95"], color="#ddcc77", alpha=0.25, label="q05-q95")
        ax.set_xlabel(key)
        ax.set_ylabel("pseudo-posterior mass")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "posterior_marginals.png", dpi=220)
    fig.savefig(out_dir / "posterior_marginals.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=ROOT)
    parser.add_argument("--round", default="corrected_ic_zdecay_anchor_000")
    parser.add_argument("--num-samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--profile-bins", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.002)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-6)
    parser.add_argument("--variance", type=float, default=0.999)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "corrected_ic_reanalysis" / "inverse_posterior_corrected_ic_dense")
    args = parser.parse_args()

    campaign_dir = args.campaign_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "corrected_ic_reanalysis" / "figures" / "generated").mkdir(parents=True, exist_ok=True)

    dataset = corrected_anchor_dataset(campaign_dir, args.round)
    model = fit_surrogate(dataset, alpha=args.ridge_alpha, variance=args.variance, max_components=args.max_components)
    lo, hi = bounds_from_config(campaign_dir)
    candidates = sample_grid(args.num_samples, lo, hi, args.seed)
    candidates = np.vstack([dataset["X"], candidates])
    rows = score_candidates(model, dataset, candidates, batch_size=args.batch_size)
    write_scores(rows, output_dir / "candidate_scores.csv")

    best = rows[0]
    observed_objectives = []
    for cand, values in zip(dataset["candidate_ids"], dataset["X"]):
        obs = dataset["observations"].get(cand, {})
        observed_objectives.append(
            {
                "candidate_id": cand,
                "g11": float(values[0]),
                "g12": float(values[1]),
                "g44": float(values[2]),
                "objective": float(obs.get("objective", np.nan)),
            }
        )
    best_observed = min(observed_objectives, key=lambda row: row["objective"])
    summary = {
        "campaign_dir": str(campaign_dir),
        "round": args.round,
        "surrogate": "POD + RBF kernel ridge regression on corrected g_ij anchor fields",
        "warning": (
            f"Dense posterior is an interpolation diagnostic trained on "
            f"{int(dataset['X'].shape[0])} corrected g_ij-only MOOSE candidates; "
            "direct MOOSE rescoring remains authoritative."
        ),
        "num_samples": int(len(rows)),
        "temperature": float(args.temperature),
        "profile_bins": int(args.profile_bins),
        "objective": "center-crop z-normalized surface uz nMSE",
        "image_shape": dataset["image_shape"],
        "condition_keys": dataset["condition_keys"],
        "surface_uz_orientation": dataset["orientation"],
        "surface_uz_edge_crop_fraction": dataset["crop_fraction"],
        "best": best,
        "best_observed": best_observed,
        "observed_anchor_objectives": observed_objectives,
        "model": {
            "training_sample_count": int(dataset["X"].shape[0]),
            "component_count": int(model["basis"].shape[0]),
            "retained_energy": float(model["retained_energy"]),
            "singular_values": [float(v) for v in model["singular_values"]],
            "explained_variance": [float(v) for v in model["explained_variance"]],
            "rbf_length_scale": float(model["length_scale"]),
            "ridge_alpha": float(model["alpha"]),
        },
        "profiles": profiles(rows, lo, hi, args.profile_bins),
        "pseudo_posterior": pseudo_posterior(rows, args.temperature),
    }
    (output_dir / "posterior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "corrected_ic_dense_pod_rbf_model.npz",
        X_train=model["X_train"],
        input_mean=model["input_mean"],
        input_scale=model["input_scale"],
        length_scale=np.asarray([model["length_scale"]]),
        alpha=np.asarray([model["alpha"]]),
        dual=model["dual"],
        coeff_mean=model["coeff_mean"],
        coeff_scale=model["coeff_scale"],
        field_mean=model["field_mean"],
        basis=model["basis"],
        singular_values=model["singular_values"],
        explained_variance=model["explained_variance"],
        condition_keys=np.asarray(dataset["condition_keys"]),
        objective_mask=np.asarray(dataset["objective_mask"], dtype=bool),
    )
    plot_loss_profiles(summary, {"best_observed": best_observed}, output_dir)
    plot_pair_projections(rows, summary, output_dir)
    plot_marginals(rows, summary, output_dir, args.temperature)
    print(json.dumps({
        "output_dir": str(output_dir),
        "loss_profiles": str(output_dir / "loss_profiles.png"),
        "best_dense": best,
        "best_observed": best_observed,
        "warning": summary["warning"],
    }, indent=2))


if __name__ == "__main__":
    main()
