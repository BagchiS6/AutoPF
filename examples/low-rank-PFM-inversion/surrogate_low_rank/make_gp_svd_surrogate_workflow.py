#!/usr/bin/env python3
"""Draw a publication-style GP/POD surrogate workflow schematic."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-simbo")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_figures" / "generated"
SUMMARY = ROOT / "gp_surrogate_combined" / "condition_field_summary.json"

COLORS = {
    "ink": "#17212b",
    "muted": "#536171",
    "light": "#f6f8fb",
    "field": "#1f9e89",
    "field_soft": "#d8f1eb",
    "pod": "#7b4ab8",
    "pod_soft": "#eadff7",
    "gp": "#2f6fbb",
    "gp_soft": "#dce9f8",
    "loss": "#c94f4f",
    "loss_soft": "#f8dfdc",
    "moose": "#d58a1f",
    "moose_soft": "#f8ead2",
    "accent": "#242a7a",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 11.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_numbers() -> dict[str, object]:
    if not SUMMARY.exists():
        return {
            "sample_count": 4769,
            "condition_count": 36,
            "pixels": 1681,
            "component_count": 25,
            "retained_energy": 0.999037,
            "explained_variance": [0.707, 0.155, 0.0628, 0.0268, 0.0141, 0.0127],
        }
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {
        "sample_count": summary.get("dataset", {}).get("sample_count", 4769),
        "condition_count": summary.get("dataset", {}).get("condition_count", 36),
        "pixels": summary.get("preprocess", {}).get("target_size", 1681),
        "component_count": summary.get("component_count", 25),
        "retained_energy": summary.get("retained_energy", 0.999037),
        "explained_variance": summary.get("explained_variance", [0.707, 0.155, 0.0628, 0.0268, 0.0141, 0.0127]),
    }


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    *,
    title: str,
    body: str,
    face: str,
    edge: str,
    title_color: str | None = None,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.02,
        y + h - 0.052,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=title_color or edge,
        fontsize=12.0,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        x + 0.02,
        y + h - 0.108,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["ink"],
        fontsize=9.45,
        linespacing=1.28,
        zorder=3,
    )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#3f4854",
    label: str | None = None,
    yoff: float = 0.025,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.45,
        color=color,
        shrinkA=4,
        shrinkB=4,
        transform=ax.transAxes,
        zorder=4,
    )
    ax.add_patch(arrow)
    if label:
        xm = 0.5 * (start[0] + end[0])
        ym = 0.5 * (start[1] + end[1]) + yoff
        ax.text(
            xm,
            ym,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=color,
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.86},
            zorder=5,
        )


def mini_field(seed: int, n: int = 70) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(xs, ys)
    cx, cy = rng.uniform(-0.12, 0.12, 2)
    rx = rng.uniform(0.22, 0.36)
    ry = rng.uniform(0.18, 0.34)
    theta = rng.uniform(-0.55, 0.55)
    xp = np.cos(theta) * (x - cx) + np.sin(theta) * (y - cy)
    yp = -np.sin(theta) * (x - cx) + np.cos(theta) * (y - cy)
    domain = np.exp(-0.5 * ((xp / rx) ** 2 + (yp / ry) ** 2))
    rim = np.exp(-0.5 * ((np.sqrt((xp / rx) ** 2 + (yp / ry) ** 2) - 1.0) / 0.16) ** 2)
    return 0.8 * domain + 0.22 * rim + 0.05 * rng.normal(size=(n, n))


def add_field_stack(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> None:
    offsets = [(0.018, 0.034), (0.009, 0.017), (0.0, 0.0)]
    for idx, (dx, dy) in enumerate(offsets):
        ax_img = fig.add_axes([left + dx, bottom + dy, width, height], zorder=6 + idx)
        arr = mini_field(10 + idx)
        ax_img.imshow(arr, cmap="viridis", origin="lower", interpolation="bicubic")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_color("white")
            spine.set_linewidth(1.1)


def add_spectrum(fig: plt.Figure, left: float, bottom: float, width: float, height: float, explained: list[float]) -> None:
    ax_spec = fig.add_axes([left, bottom, width, height], zorder=8)
    vals = np.asarray(explained[:10], dtype=float)
    ax_spec.bar(np.arange(1, len(vals) + 1), vals, color=plt.get_cmap("magma")(np.linspace(0.25, 0.82, len(vals))), width=0.72)
    ax_spec.set_yscale("log")
    ax_spec.set_xticks([1, 5, 10])
    ax_spec.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax_spec.tick_params(length=2, pad=1, labelsize=6.7)
    ax_spec.set_title("SVD spectrum", fontsize=7.8, pad=2)
    for spine in ax_spec.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#5d5267")


def main() -> None:
    configure_matplotlib()
    numbers = load_numbers()
    sample_count = int(numbers["sample_count"])
    condition_count = int(numbers["condition_count"])
    pixels = int(numbers["pixels"])
    component_count = int(numbers["component_count"])
    retained = float(numbers["retained_energy"])
    explained = [float(v) for v in numbers["explained_variance"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.2, 7.65))
    ax.set_axis_off()

    ax.text(
        0.035,
        0.962,
        "Condition-aware GP surrogate for low-rank surface-field inversion",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=19.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.035,
        0.914,
        "Learn a differentiable map from material parameters and voltage/pulse condition to the full surface observable.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.8,
        color=COLORS["muted"],
    )

    add_box(
        ax,
        (0.035, 0.62),
        (0.205, 0.21),
        title="simulation records",
        body=(
            rf"$\mathcal{{D}}=\{{(\mathbf{{x}}_i,\mathbf{{u}}_i)\}}_{{i=1}}^N$" + "\n"
            rf"$N={sample_count:,}$ fields, {condition_count} conditions" + "\n"
            rf"$\mathbf{{x}}=[g_{{11}},g_{{12}},g_{{44}},V,\log\tau]$"
        ),
        face=COLORS["field_soft"],
        edge=COLORS["field"],
    )
    add_field_stack(fig, 0.073, 0.635, 0.07, 0.10)

    add_box(
        ax,
        (0.285, 0.62),
        (0.195, 0.21),
        title="normalize + vectorize",
        body=(
            rf"$U_z(x,y)\rightarrow\mathbf{{u}}\in\mathbb{{R}}^{{{pixels}}}$" + "\n"
            r"$\mathbf{u}=\operatorname{vec}\{\operatorname{znorm}(U_z)\}$" + "\n"
            rf"$Y\in\mathbb{{R}}^{{N\times {pixels}}}$"
        ),
        face="#fff4d8",
        edge="#d59b22",
    )

    add_box(
        ax,
        (0.525, 0.62),
        (0.205, 0.21),
        title="POD / SVD basis",
        body=(
            r"$Y_c = Q\Sigma\Phi^T$" + "\n"
            r"$\mathbf{u}\approx\bar{\mathbf{u}}+\Phi_k\mathbf{a}$" + "\n"
            rf"$k={component_count}$, energy={100.0 * retained:.3f}\%"
        ),
        face=COLORS["pod_soft"],
        edge=COLORS["pod"],
    )
    add_spectrum(fig, 0.653, 0.647, 0.058, 0.088, explained)

    add_box(
        ax,
        (0.775, 0.62),
        (0.19, 0.21),
        title="multitask variational GP",
        body=(
            r"$a_j(\mathbf{x})\sim\mathcal{GP}(m_j,k_j)$" + "\n"
            r"$k_j$: ARD-RBF kernel" + "\n"
            r"$\mathbf{x}\mapsto(\boldsymbol{\mu}_a,\boldsymbol{\Sigma}_a)$"
        ),
        face=COLORS["gp_soft"],
        edge=COLORS["gp"],
    )

    add_arrow(ax, (0.242, 0.725), (0.284, 0.725), color=COLORS["field"])
    add_arrow(ax, (0.482, 0.725), (0.524, 0.725), color="#d59b22")
    add_arrow(ax, (0.732, 0.725), (0.774, 0.725), color=COLORS["pod"])

    add_box(
        ax,
        (0.095, 0.27),
        (0.245, 0.21),
        title="field decoder",
        body=(
            r"$\widehat{\mathbf{u}}(\mathbf{x})=\bar{\mathbf{u}}+\Phi_k\boldsymbol{\mu}_a(\mathbf{x})$" + "\n"
            r"$\operatorname{Var}[\widehat{\mathbf{u}}]\approx\Phi_k\boldsymbol{\Sigma}_a\Phi_k^T$" + "\n"
            r"reshape to $\widehat{U}_z(x,y;V,\tau)$"
        ),
        face="#e7f4ff",
        edge="#2487bf",
    )
    add_field_stack(fig, 0.123, 0.285, 0.075, 0.105)

    add_box(
        ax,
        (0.395, 0.27),
        (0.235, 0.21),
        title="inverse objective",
        body=(
            r"$\mathcal{L}(\mathbf{g})=$" + "\n"
            r"$\frac{1}{|\mathcal{C}|}\sum_{(V,\tau)\in\mathcal{C}}"
            r"\frac{\|M\Delta U_z\|_2^2}{\|M U_z^{\rm PFM}\|_2^2}$" + "\n"
            r"$\Delta U_z=\widehat{U}_z-U_z^{\rm PFM}$" + "\n"
            r"$M$: center crop / edge mask"
        ),
        face=COLORS["loss_soft"],
        edge=COLORS["loss"],
    )

    add_box(
        ax,
        (0.685, 0.27),
        (0.24, 0.21),
        title="acquire + rescore",
        body=(
            r"$A(\mathbf{g})=\widehat{\mathcal{L}}(\mathbf{g})-\kappa\sqrt{\overline{\sigma_f^2}}$" + "\n"
            r"propose candidates by GP-UQ" + "\n"
            r"accept only after direct MOOSE"
        ),
        face=COLORS["moose_soft"],
        edge=COLORS["moose"],
    )

    add_arrow(ax, (0.87, 0.617), (0.245, 0.483), color=COLORS["gp"], label=r"predict field + uncertainty", yoff=0.018)
    add_arrow(ax, (0.342, 0.374), (0.394, 0.374), color="#2487bf", label=r"$\widehat{U}_z$", yoff=0.052)
    add_arrow(ax, (0.632, 0.374), (0.684, 0.374), color=COLORS["loss"])

    loop = FancyArrowPatch(
        (0.805, 0.268),
        (0.14, 0.617),
        connectionstyle="arc3,rad=-0.36",
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.45,
        linestyle=(0, (5, 3)),
        color=COLORS["moose"],
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(loop)
    ax.text(
        0.465,
        0.105,
        "Direct MOOSE/AutoPF results are appended to the dataset; the surrogate is a proposal engine, not the final acceptor.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.0,
        color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.42,rounding_size=0.018", "fc": COLORS["light"], "ec": "#d4dae4"},
    )

    ax.text(
        0.035,
        0.045,
        "Key distinction: scalar inputs are low-dimensional, but each observation is a full field; POD makes the GP learn coefficient maps rather than pixels.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.6,
        color=COLORS["ink"],
    )

    pdf = OUT_DIR / "fig_gp_svd_surrogate_workflow.pdf"
    png = OUT_DIR / "fig_gp_svd_surrogate_workflow.png"
    manifest = OUT_DIR / "fig_gp_svd_surrogate_workflow_manifest.json"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    manifest.write_text(
        json.dumps(
            {
                "pdf": str(pdf),
                "png": str(png),
                "source_summary": str(SUMMARY),
                "sample_count": sample_count,
                "condition_count": condition_count,
                "pixel_count": pixels,
                "component_count": component_count,
                "retained_energy": retained,
                "description": "Color schematic of the condition-aware GP surrogate with POD/SVD compression.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pdf)
    print(png)
    print(manifest)


if __name__ == "__main__":
    main()
