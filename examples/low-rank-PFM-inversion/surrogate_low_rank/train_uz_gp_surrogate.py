#!/usr/bin/env python3
"""Train a variational multitask GP surrogate for low-rank ``uz`` fields.

This script uses the data extraction and POD/SVD machinery from
``train_uz_svd_surrogate.py``, then fits a GPyTorch variational GP that maps
inputs to the retained SVD coefficient vector.

Modes:

* candidate-grid: [g11, g12, g44] -> coefficients for the fixed condition grid.
* condition-field: [g11, g12, g44, tip_voltage, pulse_end] -> coefficients for
  one surface field.

The GP treats each retained SVD coefficient as an output task using
``IndependentMultitaskVariationalStrategy``. This is a pragmatic first model:
it scales to the current dataset and gives predictive uncertainty in SVD
coefficient space.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import gpytorch

import train_uz_svd_surrogate as svd


class IndependentMultitaskSVDDKL(gpytorch.models.ApproximateGP):
    """Batch-independent variational GP, wrapped as multitask outputs."""

    def __init__(self, inducing_points: torch.Tensor, num_tasks: int) -> None:
        batch_shape = torch.Size([num_tasks])
        if inducing_points.dim() == 2:
            inducing_points = inducing_points.unsqueeze(0).expand(num_tasks, -1, -1).contiguous()

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=batch_shape,
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
        )
        super().__init__(strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=inducing_points.size(-1),
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def torch_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_dataset(args: argparse.Namespace):
    campaign_dir = Path(args.campaign_dir).resolve()
    config = svd.load_json(campaign_dir / "sim_bo_config.json")
    source_dirs = [Path(p).resolve() for p in args.source_campaign_dir]
    if not source_dirs:
        source_dirs = [campaign_dir]

    records = []
    image_shape = None
    collection_summary = {
        "source_campaign_dirs": [str(p) for p in source_dirs],
        "source_summaries": [],
        "missing_output_count": 0,
        "missing_outputs_first10": [],
        "surface_key_counts": {},
        "condition_counts": {},
    }
    for source_dir in source_dirs:
        source_records, source_shape, source_summary = svd.collect_records(
            source_dir,
            field=args.field,
            normalization=args.normalization,
            include_holdout=args.include_holdout,
            exclude_rounds=args.exclude_round,
        )
        if image_shape is None:
            image_shape = source_shape
        elif tuple(image_shape) != tuple(source_shape):
            raise ValueError(
                f"Shape mismatch for {source_dir}: {source_shape} != {image_shape}"
            )
        records.extend(source_records)
        collection_summary["source_summaries"].append(
            {"campaign_dir": str(source_dir), **source_summary}
        )
        collection_summary["missing_output_count"] += source_summary.get("missing_output_count", 0)
        collection_summary["missing_outputs_first10"].extend(
            source_summary.get("missing_outputs_first10", [])
        )
        collection_summary["missing_outputs_first10"] = collection_summary["missing_outputs_first10"][:10]
        for key, value in source_summary.get("surface_key_counts", {}).items():
            collection_summary["surface_key_counts"][key] = (
                collection_summary["surface_key_counts"].get(key, 0) + value
            )
        for key, value in source_summary.get("condition_counts", {}).items():
            collection_summary["condition_counts"][key] = (
                collection_summary["condition_counts"].get(key, 0) + value
            )

    if args.mode == "candidate-grid":
        X, Y, groups, input_names, dataset_summary = svd.build_candidate_grid_dataset(
            records,
            config,
            args.condition_set,
        )
    elif args.mode == "condition-field":
        X, Y, groups, input_names, dataset_summary = svd.build_condition_field_dataset(records)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    train_idx, test_idx = svd.group_split(groups, args.test_fraction, args.random_seed)
    if train_idx.size == 0:
        raise RuntimeError("Empty training split")

    X_scaled, input_mean, input_scale = svd.standardize_inputs(X[train_idx], X)
    field_mean, basis, singular_values, explained, retained_energy, coeff_train = svd.compute_pod(
        Y[train_idx],
        n_components=args.n_components,
        variance=args.variance,
        max_components=args.max_components,
    )
    coeff = np.dot(Y - field_mean, basis.T)
    coeff_mean = np.mean(coeff_train, axis=0)
    coeff_scale = np.std(coeff_train, axis=0)
    coeff_scale[coeff_scale <= 1.0e-12] = 1.0
    coeff_scaled = (coeff - coeff_mean) / coeff_scale

    metadata = {
        "campaign_dir": str(campaign_dir),
        "mode": args.mode,
        "field": args.field,
        "normalization": args.normalization,
        "include_holdout": bool(args.include_holdout),
        "exclude_round": list(args.exclude_round),
        "condition_set": args.condition_set if args.mode == "candidate-grid" else None,
        "image_shape": list(image_shape),
        "input_names": input_names,
        "sample_count": int(X.shape[0]),
        "target_dimension": int(Y.shape[1]),
        "train_sample_count": int(train_idx.size),
        "test_sample_count": int(test_idx.size),
        "group_count": int(len(set(groups.tolist()))),
        "component_count": int(basis.shape[0]),
        "retained_energy": float(retained_energy),
        "singular_values": singular_values.tolist(),
        "explained_variance": explained.tolist(),
        "dataset": dataset_summary,
        "collection": collection_summary,
    }
    arrays = {
        "X": X,
        "Y": Y,
        "groups": groups,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_scaled": X_scaled,
        "input_mean": input_mean,
        "input_scale": input_scale,
        "coeff_scaled": coeff_scaled,
        "coeff_mean": coeff_mean,
        "coeff_scale": coeff_scale,
        "field_mean": field_mean,
        "basis": basis,
        "singular_values": singular_values,
        "explained_variance": explained,
        "condition_keys": np.asarray(dataset_summary.get("condition_keys", [])),
    }
    return arrays, metadata


def choose_inducing(train_x: torch.Tensor, num_inducing: int, seed: int) -> torch.Tensor:
    n = train_x.size(0)
    m = min(int(num_inducing), n)
    generator = torch.Generator(device=train_x.device)
    generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator, device=train_x.device)
    return train_x[perm[:m]].contiguous()


def reconstruct_error(y_true: np.ndarray, coeff_pred_scaled: np.ndarray, arrays: dict) -> dict:
    coeff_pred = coeff_pred_scaled * arrays["coeff_scale"] + arrays["coeff_mean"]
    pred = arrays["field_mean"] + np.dot(coeff_pred, arrays["basis"])
    return svd.summarize_errors(y_true, pred, arrays["field_mean"])


def coefficient_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sqrt(np.mean(y_true ** 2)))
    rel = rmse / denom if denom > 1.0e-12 else float("nan")
    return {"rmse": rmse, "mae": mae, "relative_rmse": rel}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = select_device(args.device)
    dtype = torch_dtype(args.dtype)

    arrays, summary = make_dataset(args)
    train_idx = arrays["train_idx"]
    test_idx = arrays["test_idx"]

    train_x = torch.as_tensor(arrays["X_scaled"][train_idx], dtype=dtype, device=device)
    train_y = torch.as_tensor(arrays["coeff_scaled"][train_idx], dtype=dtype, device=device)
    all_x = torch.as_tensor(arrays["X_scaled"], dtype=dtype, device=device)

    inducing = choose_inducing(train_x, args.num_inducing, args.random_seed)
    model = IndependentMultitaskSVDDKL(inducing, num_tasks=train_y.size(-1)).to(device=device, dtype=dtype)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=train_y.size(-1),
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=args.learning_rate,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    dataset = TensorDataset(train_x, train_y)
    batch_size = min(int(args.batch_size), train_x.size(0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    likelihood.train()
    loss_history = []
    t0 = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        seen = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            batch_n = xb.size(0)
            epoch_loss += float(loss.detach().cpu()) * batch_n
            seen += batch_n

        epoch_loss /= max(seen, 1)
        loss_history.append(epoch_loss)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(f"epoch {epoch:04d}/{args.epochs}  loss={epoch_loss:.6f}", flush=True)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(all_x))
        pred_mean = pred_dist.mean.detach().cpu().numpy()
        pred_var = pred_dist.variance.detach().cpu().numpy()

    y_coeff = arrays["coeff_scaled"]
    y_field = arrays["Y"]
    train_coeff_errors = coefficient_error(y_coeff[train_idx], pred_mean[train_idx])
    train_field_errors = reconstruct_error(y_field[train_idx], pred_mean[train_idx], arrays)
    test_coeff_errors = None
    test_field_errors = None
    if test_idx.size:
        test_coeff_errors = coefficient_error(y_coeff[test_idx], pred_mean[test_idx])
        test_field_errors = reconstruct_error(y_field[test_idx], pred_mean[test_idx], arrays)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.campaign_dir).resolve() / "gp_surrogate"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.mode.replace("-", "_")
    gp_path = output_dir / f"{stem}_gp_state.pt"
    metadata_path = output_dir / f"{stem}_preprocess.npz"
    summary_path = output_dir / f"{stem}_summary.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "input_dim": int(train_x.size(-1)),
            "num_tasks": int(train_y.size(-1)),
            "num_inducing": int(inducing.size(-2)),
            "dtype": args.dtype,
            "mode": args.mode,
            "inducing_points": inducing.detach().cpu(),
        },
        gp_path,
    )

    np.savez_compressed(
        metadata_path,
        mode=np.asarray([args.mode]),
        field=np.asarray([args.field]),
        normalization=np.asarray([args.normalization]),
        input_names=np.asarray(summary["input_names"]),
        image_shape=np.asarray(summary["image_shape"], dtype=np.int64),
        input_mean=arrays["input_mean"],
        input_scale=arrays["input_scale"],
        coeff_mean=arrays["coeff_mean"],
        coeff_scale=arrays["coeff_scale"],
        field_mean=arrays["field_mean"],
        basis=arrays["basis"],
        singular_values=arrays["singular_values"],
        explained_variance=arrays["explained_variance"],
        X=arrays["X"],
        groups=arrays["groups"],
        train_indices=train_idx,
        test_indices=test_idx,
        condition_keys=arrays["condition_keys"],
        predictive_coeff_mean=pred_mean,
        predictive_coeff_variance=pred_var,
    )

    summary.update(
        {
            "gp": {
                "model": "IndependentMultitaskSVDDKL",
                "library": {
                    "torch": torch.__version__,
                    "gpytorch": gpytorch.__version__,
                },
                "device": str(device),
                "dtype": args.dtype,
                "epochs": int(args.epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(args.learning_rate),
                "num_inducing": int(inducing.size(-2)),
                "loss_history": loss_history,
                "wall_time_seconds": float(time.monotonic() - t0),
            },
            "train_coefficient_errors": train_coeff_errors,
            "test_coefficient_errors": test_coeff_errors,
            "train_field_errors": train_field_errors,
            "test_field_errors": test_field_errors,
            "artifacts": {
                "gp_state": str(gp_path),
                "preprocess": str(metadata_path),
                "summary": str(summary_path),
            },
        }
    )
    summary = jsonable(summary)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary


def main() -> None:
    default_campaign = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", default=str(default_campaign))
    parser.add_argument(
        "--source-campaign-dir",
        action="append",
        default=[],
        help=(
            "Campaign directory to use for training records. May be repeated. "
            "Defaults to --campaign-dir when omitted."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("candidate-grid", "condition-field"),
        default="candidate-grid",
    )
    parser.add_argument("--field", default="disp_z")
    parser.add_argument(
        "--normalization",
        choices=("znorm", "center", "raw"),
        default="znorm",
    )
    parser.add_argument(
        "--condition-set",
        choices=("common", "train", "all"),
        default="common",
    )
    parser.add_argument("--exclude-holdout", dest="include_holdout", action="store_false")
    parser.set_defaults(include_holdout=True)
    parser.add_argument(
        "--exclude-round",
        action="append",
        default=[],
        help="Round directory name to omit from training, e.g. round_anchors. May be repeated.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--n-components", type=int, default=0)
    parser.add_argument("--variance", type=float, default=0.999)
    parser.add_argument("--max-components", type=int, default=32)
    parser.add_argument("--num-inducing", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    try:
        summary = train(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    print("Trained GP surrogate")
    print(f"  mode: {summary['mode']}")
    print(f"  samples: {summary['train_sample_count']} train / {summary['test_sample_count']} test")
    print(f"  components: {summary['component_count']} retained_energy={summary['retained_energy']:.6f}")
    print(f"  device: {summary['gp']['device']}")
    print(f"  train field relative_rmse: {summary['train_field_errors']['relative_rmse']:.6f}")
    if summary["test_field_errors"] is not None:
        print(f"  test field relative_rmse: {summary['test_field_errors']['relative_rmse']:.6f}")
    print(f"  summary: {summary['artifacts']['summary']}")


if __name__ == "__main__":
    main()
