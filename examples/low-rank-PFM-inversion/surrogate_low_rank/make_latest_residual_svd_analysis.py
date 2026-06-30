#!/usr/bin/env python3
"""Residual-SVD diagnostics for the latest anisotropic hidden-physics holdout.

The residual definition intentionally matches the holdout loss:

  1. load final MOOSE surface ``disp_z``/``uz``;
  2. apply the manifest orientation correction;
  3. resample to the experimental PFM-u_z grid;
  4. z-normalize simulation and experiment independently;
  5. compute residual = simulation - experiment;
  6. crop the edge fraction used by the objective.

This makes the SVD spectrum directly comparable to the reported center-crop
nMSE values in the latest hidden-physics discovery figures.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simbo")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
OUT = HERE / "generated"
TODO = HERE / "needs_regeneration_for the_latest_campaign"
ANALYSIS = CAMPAIGN / "anisotropic_hidden_analysis"
ROUND_NAME = "anisotropic_holdout_full_000"
ROUND_DIR = CAMPAIGN / f"round_{ROUND_NAME}"
SUMMARY_DIR = ANALYSIS / ROUND_NAME

sys.path.insert(0, str(CAMPAIGN))
from surface_orientation import orient_sim_for_experiment, resample_array  # noqa: E402


COLORS = {
    "hidden": "#8A8A8A",
    "screening": "#E45756",
    "flexo": "#F58518",
    "anis": "#54A24B",
    "dark": "#222222",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.titlesize": 11.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def znorm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    std = float(np.nanstd(arr))
    return (arr - float(np.nanmean(arr))) / (std if std > 1.0e-12 else 1.0)


def crop(arr: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return arr
    ny, nx = arr.shape
    dy = int(round(fraction * ny))
    dx = int(round(fraction * nx))
    if dy * 2 >= ny or dx * 2 >= nx:
        return arr
    return arr[dy : ny - dy, dx : nx - dx]


def candidate_color(name: str) -> str:
    if "hidden_off" in name:
        return COLORS["hidden"]
    if "screen" in name:
        return COLORS["screening"]
    if "flexo" in name:
        return COLORS["flexo"]
    return COLORS["anis"]


def short_name(name: str) -> str:
    mapping = {
        "fixed_g_hidden_off": "hidden off",
        "screening_only_refined_control": "screening only",
        "flexo_proxy_grad_neg": "flexo proxy",
        "shear_yz_pos": r"anis. $\epsilon_{yz}$",
    }
    return mapping.get(name, name.replace("_", " "))


def format_condition(run: dict[str, Any]) -> str:
    return f"{float(run['tip_voltage']):g} V, {float(run['pulse_end']):g} s"


def load_residual_stack(manifest: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    orientation = str(manifest.get("loss", {}).get("surface_uz_orientation", "identity"))
    crop_fraction = float(manifest.get("loss", {}).get("surface_uz_edge_crop_fraction", 0.0))
    runs = [
        run
        for run in manifest["runs"]
        if str(run.get("candidate_id")) == str(candidate_id)
    ]
    runs = sorted(runs, key=lambda row: (float(row["tip_voltage"]), float(row["pulse_end"])))
    residuals: list[np.ndarray] = []
    for run in runs:
        fields_path = Path(run["output_dir"]) / "fields_final_timestep.npz"
        fields = np.load(fields_path, allow_pickle=False)
        sim_key = "disp_z" if "disp_z" in fields.files else "uz"
        sim = orient_sim_for_experiment(np.asarray(fields[sim_key], dtype=float), orientation)
        exp = np.asarray(np.load(run["experimental_uz_path"], allow_pickle=False)["uz"], dtype=float)
        sim = resample_array(sim, exp.shape)
        resid = znorm(sim) - znorm(exp)
        residuals.append(crop(resid, crop_fraction))
    matrix = np.vstack([arr.ravel() for arr in residuals])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)
    energy = svals**2
    total = float(np.sum(energy))
    frac = energy / total if total > 0 else np.zeros_like(energy)
    cumulative = np.cumsum(frac)
    return {
        "candidate_id": candidate_id,
        "runs": runs,
        "residuals": residuals,
        "matrix": matrix,
        "centered": centered,
        "shape": residuals[0].shape,
        "singular_values": svals,
        "variance_fraction": frac,
        "cumulative_variance": cumulative,
        "modes": vt,
        "n90": int(np.searchsorted(cumulative, 0.90) + 1) if len(cumulative) else 0,
        "n95": int(np.searchsorted(cumulative, 0.95) + 1) if len(cumulative) else 0,
    }


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def plot_spectrum(ax: plt.Axes, svd: dict[str, Any], title: str, color: str) -> None:
    frac = np.asarray(svd["variance_fraction"], dtype=float)
    cumulative = np.asarray(svd["cumulative_variance"], dtype=float)
    n = min(8, len(frac))
    modes = np.arange(1, n + 1)
    ax.bar(modes, 100.0 * frac[:n], color=color, edgecolor="white", linewidth=0.7, alpha=0.85)
    ax.plot(modes, 100.0 * cumulative[:n], color=COLORS["dark"], marker="o", lw=1.2, ms=3.2)
    ax.set_ylim(0, 105)
    ax.set_xticks(modes)
    ax.set_xlabel("mode")
    ax.set_ylabel("variance (%)")
    ax.set_title(title)
    ax.grid(axis="y", color="#D8D8D8", lw=0.65, alpha=0.8)
    ax.text(
        0.98,
        0.08,
        rf"$N_{{90}}={svd['n90']}$" + "\n" + rf"$N_{{95}}={svd['n95']}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
    )


def save(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    TODO.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext, dpi in (("png", 450), ("pdf", None)):
        path = OUT / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=dpi)
        else:
            fig.savefig(path)
        paths[ext] = str(path)
        target = TODO / path.name
        target.write_bytes(path.read_bytes())
    plt.close(fig)
    return paths


def make_mode_figure(
    ranking: list[dict[str, str]],
    spectra: dict[str, dict[str, Any]],
) -> dict[str, str]:
    hidden = next(row for row in ranking if row["name"] == "fixed_g_hidden_off")
    best = ranking[0]
    selected = [hidden, best]
    n_modes = 4
    mode_arrays = []
    for row in selected:
        svd = spectra[row["candidate_id"]]
        for idx in range(n_modes):
            mode_arrays.append(svd["modes"][idx].reshape(svd["shape"]))
    vmax = max(float(np.nanpercentile(np.abs(np.concatenate([a.ravel() for a in mode_arrays])), 99.2)), 1.0e-9)

    fig = plt.figure(figsize=(12.8, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, n_modes + 1, width_ratios=[1.25, 1, 1, 1, 1])
    letters = ["a", "b"]
    for row_idx, row in enumerate(selected):
        name = row["name"]
        color = candidate_color(name)
        svd = spectra[row["candidate_id"]]
        ax_spec = fig.add_subplot(gs[row_idx, 0])
        plot_spectrum(ax_spec, svd, f"{short_name(name)} residual spectrum", color)
        panel(ax_spec, letters[row_idx])
        for mode_idx in range(n_modes):
            ax = fig.add_subplot(gs[row_idx, mode_idx + 1])
            mode = svd["modes"][mode_idx].reshape(svd["shape"])
            im = ax.imshow(mode, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"mode {mode_idx + 1}\n{100.0 * svd['variance_fraction'][mode_idx]:.1f}%")
            ax.set_xticks([])
            ax.set_yticks([])
            if mode_idx == n_modes - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_label("SVD loading")
    fig.suptitle("Residual SVD before and after hidden-physics correction", fontweight="bold")
    return save(fig, "fig_latest_hidden_physics_residual_svd_modes")


def make_all_spectra_figure(
    ranking: list[dict[str, str]],
    spectra: dict[str, dict[str, Any]],
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    ax = axes[0]
    for row in ranking:
        svd = spectra[row["candidate_id"]]
        frac = np.asarray(svd["variance_fraction"], dtype=float)
        n = min(8, len(frac))
        ax.plot(
            np.arange(1, n + 1),
            100.0 * np.cumsum(frac[:n]),
            marker="o",
            lw=1.5,
            ms=3,
            color=candidate_color(row["name"]),
            label=short_name(row["name"]),
        )
    ax.set_xlabel("residual SVD mode")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_ylim(0, 105)
    ax.set_title("low-rank residual structure")
    ax.grid(color="#D8D8D8", lw=0.65, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    panel(ax, "a")

    ax = axes[1]
    x = np.arange(len(ranking))
    n90 = [spectra[row["candidate_id"]]["n90"] for row in ranking]
    n95 = [spectra[row["candidate_id"]]["n95"] for row in ranking]
    width = 0.36
    ax.bar(x - width / 2, n90, width, color="#4C78A8", label=r"$N_{90}$")
    ax.bar(x + width / 2, n95, width, color="#F58518", label=r"$N_{95}$")
    ax.set_xticks(x, [short_name(row["name"]) for row in ranking], rotation=25, ha="right")
    ax.set_ylabel("modes required")
    ax.set_title("rank needed to capture residual variance")
    ax.grid(axis="y", color="#D8D8D8", lw=0.65, alpha=0.8)
    ax.legend(frameon=False)
    for xi, a, b in zip(x, n90, n95):
        ax.text(xi - width / 2, a + 0.12, str(a), ha="center", va="bottom", fontsize=7)
        ax.text(xi + width / 2, b + 0.12, str(b), ha="center", va="bottom", fontsize=7)
    panel(ax, "b")

    fig.suptitle("Residual rank diagnostic across hidden-physics candidates", fontweight="bold")
    return save(fig, "fig_latest_hidden_physics_residual_svd_rank_summary")


def main() -> None:
    configure()
    manifest = load_json(ROUND_DIR / "manifest.json")
    ranking = read_csv(SUMMARY_DIR / "candidate_ranking.csv")
    ranking = sorted(ranking, key=lambda row: float(row["objective"]))

    spectra: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    for row in ranking:
        svd = load_residual_stack(manifest, row["candidate_id"])
        spectra[row["candidate_id"]] = svd
        summary[row["candidate_id"]] = {
            "name": row["name"],
            "objective": float(row["objective"]),
            "n_conditions": int(row["n_conditions"]),
            "singular_values": [float(v) for v in svd["singular_values"]],
            "variance_fraction": [float(v) for v in svd["variance_fraction"]],
            "cumulative_variance": [float(v) for v in svd["cumulative_variance"]],
            "n90": int(svd["n90"]),
            "n95": int(svd["n95"]),
            "residual_shape_after_crop": list(svd["shape"]),
        }

    mode_paths = make_mode_figure(ranking, spectra)
    rank_paths = make_all_spectra_figure(ranking, spectra)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary_path = ANALYSIS / "latest_hidden_physics_residual_svd_summary.json"
    manifest_out = {
        "round": ROUND_NAME,
        "source_manifest": str(ROUND_DIR / "manifest.json"),
        "source_ranking": str(SUMMARY_DIR / "candidate_ranking.csv"),
        "residual_definition": "transpose-corrected z-normalized surface disp_z minus experimental uz, center-cropped by manifest edge fraction",
        "figures": {
            "mode_figure": mode_paths,
            "rank_summary": rank_paths,
        },
        "spectra": summary,
        "note": "SVD mode signs are arbitrary; compare spatial structure and variance fractions, not absolute sign.",
    }
    summary_path.write_text(json.dumps(manifest_out, indent=2) + "\n", encoding="utf-8")
    for directory in (OUT, TODO):
        (directory / "fig_latest_hidden_physics_residual_svd_manifest.json").write_text(
            json.dumps(manifest_out, indent=2) + "\n", encoding="utf-8"
        )

    print(mode_paths["pdf"])
    print(mode_paths["png"])
    print(rank_paths["pdf"])
    print(rank_paths["png"])
    print(summary_path)


if __name__ == "__main__":
    main()
