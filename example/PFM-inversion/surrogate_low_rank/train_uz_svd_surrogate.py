#!/usr/bin/env python3
"""Train a NumPy-only SVD/POD surrogate for simulated surface ``uz``.

The BO loop stores the observable as ``disp_z`` in each
``fields_final_timestep.npz``. This script walks the campaign manifests,
builds a low-rank SVD/POD representation of those surfaces, and fits a
ridge-style coefficient model to predict the leading SVD coefficients.

Two modes are supported:

* candidate-grid: X = [g11, g12, g44], Y = concatenated surfaces over a fixed
  set of voltage/pulse conditions. This is the direct g -> observable-grid map.
* condition-field: X = [g11, g12, g44, tip_voltage, pulse_end], Y = one surface.
  This can interpolate both in material parameters and in pulse conditions.

Only NumPy is required, which keeps the workflow usable on the current NERSC
environment where sklearn/torch/scipy are not available.
"""

from __future__ import print_function

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SURFACE_KEYS = ("disp_z", "uz", "polar_z")
PARAM_KEYS = ("g11", "g12", "g44")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def as_float(value):
    return float(value)


def load_surface(npz_path, field):
    with np.load(str(npz_path), allow_pickle=False) as data:
        if field != "auto":
            if field not in data.files:
                raise KeyError("Field {0!r} not found in {1}".format(field, npz_path))
            arr = data[field]
        else:
            arr = None
            for key in SURFACE_KEYS:
                if key in data.files:
                    arr = data[key]
                    break
            if arr is None:
                raise KeyError("No recognized surface field in {0}".format(npz_path))
        return np.asarray(arr, dtype=np.float64)


def normalize_field(arr, mode):
    arr = np.asarray(arr, dtype=np.float64)
    if mode == "raw":
        return arr
    if mode == "center":
        return arr - float(np.mean(arr))
    if mode == "znorm":
        centered = arr - float(np.mean(arr))
        scale = float(np.std(centered))
        if scale <= 1.0e-12:
            scale = 1.0
        return centered / scale
    raise ValueError("Unknown field normalization: {0}".format(mode))


def manifest_sort_key(path):
    name = Path(path).parent.name
    if name == "round_holdout":
        return (10 ** 9, name)
    try:
        return (int(name.split("_", 1)[1]), name)
    except Exception:
        return (10 ** 8, name)


def config_condition_order(config):
    ordered = []
    seen = set()
    for ref in config.get("train_refs", []) + config.get("holdout_refs", []):
        key = ref["key"]
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def collect_records(campaign_dir, field, normalization, include_holdout, exclude_rounds=None):
    campaign_dir = Path(campaign_dir)
    exclude_rounds = set(exclude_rounds or [])
    manifest_paths = sorted(
        campaign_dir.glob("round_*/manifest.json"),
        key=manifest_sort_key,
    )
    records = []
    shape = None
    missing = []
    condition_counts = Counter()
    field_key_counts = Counter()

    for manifest_path in manifest_paths:
        if manifest_path.parent.name in exclude_rounds:
            continue
        if manifest_path.parent.name == "round_holdout" and not include_holdout:
            continue
        manifest = load_json(manifest_path)
        for run in manifest.get("runs", []):
            npz_path = Path(run["output_dir"]) / "fields_final_timestep.npz"
            if not npz_path.exists():
                missing.append(str(npz_path))
                continue

            arr = load_surface(npz_path, field)
            if shape is None:
                shape = tuple(arr.shape)
            elif tuple(arr.shape) != shape:
                raise ValueError(
                    "Inconsistent shape {0} in {1}; expected {2}".format(
                        tuple(arr.shape), npz_path, shape
                    )
                )

            with np.load(str(npz_path), allow_pickle=False) as data:
                for key in SURFACE_KEYS:
                    if key in data.files:
                        field_key_counts[key] += 1
                        break

            norm_arr = normalize_field(arr, normalization)
            params = run.get("params") or {
                key: run[key] for key in PARAM_KEYS if key in run
            }
            param_values = tuple(as_float(params[key]) for key in PARAM_KEYS)
            condition_key = run["experiment_key"]
            condition_counts[condition_key] += 1
            records.append(
                {
                    "round": manifest_path.parent.name,
                    "candidate_id": run["candidate_id"],
                    "condition_key": condition_key,
                    "tip_voltage": as_float(run["tip_voltage"]),
                    "pulse_end": as_float(run["pulse_end"]),
                    "params": {key: as_float(params[key]) for key in PARAM_KEYS},
                    "param_values": param_values,
                    "field": norm_arr.reshape(-1),
                    "path": str(npz_path),
                }
            )

    if not records:
        raise RuntimeError("No surface records were collected from {0}".format(campaign_dir))

    return records, shape, {
        "manifest_count": len(manifest_paths),
        "missing_output_count": len(missing),
        "missing_outputs_first10": missing[:10],
        "condition_counts": dict(condition_counts),
        "surface_key_counts": dict(field_key_counts),
    }


def param_tuple_key(values, ndigits=12):
    return tuple(round(float(v), ndigits) for v in values)


def build_candidate_grid_dataset(records, config, condition_set):
    by_param = defaultdict(dict)
    params_by_key = {}
    for record in records:
        key = param_tuple_key(record["param_values"])
        by_param[key][record["condition_key"]] = record["field"]
        params_by_key[key] = record["param_values"]

    config_order = config_condition_order(config)
    observed_conditions = sorted({r["condition_key"] for r in records})

    if condition_set == "train":
        selected = [ref["key"] for ref in config.get("train_refs", [])]
    elif condition_set == "all":
        selected = [key for key in config_order if key in observed_conditions]
        if not selected:
            selected = observed_conditions
    elif condition_set == "common":
        condition_sets = [set(v.keys()) for v in by_param.values()]
        common = set.intersection(*condition_sets)
        selected = [key for key in config_order if key in common]
        selected += sorted(common.difference(selected))
    else:
        raise ValueError("Unknown condition set: {0}".format(condition_set))

    if not selected:
        raise RuntimeError("No conditions selected for candidate-grid dataset")

    rows = []
    targets = []
    kept_param_keys = []
    skipped = 0
    for key in sorted(by_param):
        condition_map = by_param[key]
        if not all(cond in condition_map for cond in selected):
            skipped += 1
            continue
        rows.append(params_by_key[key])
        targets.append(np.concatenate([condition_map[cond] for cond in selected]))
        kept_param_keys.append(key)

    if not rows:
        raise RuntimeError(
            "No parameter groups have all selected conditions: {0}".format(selected)
        )

    X = np.asarray(rows, dtype=np.float64)
    Y = np.vstack(targets).astype(np.float64)
    groups = np.asarray(["|".join(map(str, key)) for key in kept_param_keys])
    metadata = {
        "sample_count": int(X.shape[0]),
        "skipped_parameter_groups": int(skipped),
        "condition_keys": selected,
        "condition_count": int(len(selected)),
        "target_layout": "condition-major flattened surfaces",
    }
    return X, Y, groups, list(PARAM_KEYS), metadata


def build_condition_field_dataset(records):
    rows = []
    targets = []
    groups = []
    for record in records:
        rows.append(
            list(record["param_values"])
            + [record["tip_voltage"], record["pulse_end"]]
        )
        targets.append(record["field"])
        groups.append("|".join(map(str, param_tuple_key(record["param_values"]))))
    X = np.asarray(rows, dtype=np.float64)
    Y = np.vstack(targets).astype(np.float64)
    metadata = {
        "sample_count": int(X.shape[0]),
        "condition_count": int(len(set(r["condition_key"] for r in records))),
        "target_layout": "single flattened surface per record",
    }
    return X, Y, np.asarray(groups), list(PARAM_KEYS) + ["tip_voltage", "pulse_end"], metadata


def group_split(groups, test_fraction, random_seed):
    unique = np.asarray(sorted(set(groups.tolist() if hasattr(groups, "tolist") else groups)))
    if len(unique) < 2 or test_fraction <= 0.0:
        all_idx = np.arange(len(groups))
        return all_idx, np.asarray([], dtype=int)

    rng = np.random.RandomState(random_seed)
    rng.shuffle(unique)
    n_test = int(round(float(len(unique)) * test_fraction))
    n_test = max(1, min(n_test, len(unique) - 1))
    test_groups = set(unique[:n_test].tolist())
    test_mask = np.asarray([g in test_groups for g in groups], dtype=bool)
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


def monomial_powers(n_inputs, degree):
    powers = []

    def rec(pos, remaining, current):
        if pos == n_inputs:
            if remaining == 0:
                powers.append(tuple(current))
            return
        for exponent in range(remaining + 1):
            current.append(exponent)
            rec(pos + 1, remaining - exponent, current)
            current.pop()

    for total_degree in range(degree + 1):
        rec(0, total_degree, [])
    return powers


def polynomial_features(X, powers):
    Phi = np.ones((X.shape[0], len(powers)), dtype=np.float64)
    for col, power in enumerate(powers):
        values = np.ones(X.shape[0], dtype=np.float64)
        for dim, exponent in enumerate(power):
            if exponent:
                values *= X[:, dim] ** exponent
        Phi[:, col] = values
    return Phi


def standardize_inputs(X_train, X_all):
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale[scale <= 1.0e-12] = 1.0
    return (X_all - mean) / scale, mean, scale


def fit_ridge(Phi, Y, alpha):
    reg = np.eye(Phi.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    lhs = np.dot(Phi.T, Phi) + reg
    rhs = np.dot(Phi.T, Y)
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def squared_distances(A, B):
    A2 = np.sum(A * A, axis=1).reshape(-1, 1)
    B2 = np.sum(B * B, axis=1).reshape(1, -1)
    D2 = A2 + B2 - 2.0 * np.dot(A, B.T)
    return np.maximum(D2, 0.0)


def rbf_kernel(A, B, length_scale):
    D2 = squared_distances(A, B)
    return np.exp(-0.5 * D2 / (float(length_scale) ** 2))


def median_pairwise_distance(X):
    if X.shape[0] < 2:
        return 1.0
    D2 = squared_distances(X, X)
    tri = np.triu_indices(X.shape[0], k=1)
    distances = np.sqrt(D2[tri])
    distances = distances[distances > 1.0e-12]
    if distances.size == 0:
        return 1.0
    return float(np.median(distances))


def fit_rbf_krr(X_train, Y_train, alpha, length_scale):
    if not length_scale or length_scale <= 0.0:
        length_scale = median_pairwise_distance(X_train)
    K = rbf_kernel(X_train, X_train, length_scale)
    K += np.eye(K.shape[0], dtype=np.float64) * float(alpha)
    try:
        dual = np.linalg.solve(K, Y_train)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(K, Y_train, rcond=None)[0]
    return dual, float(length_scale)


def compute_pod(Y_train, n_components, variance, max_components):
    mean = np.mean(Y_train, axis=0)
    centered = Y_train - mean
    n_samples, n_features = centered.shape

    if n_samples <= n_features:
        gram = np.dot(centered, centered.T)
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        left = eigvecs[:, order]
        singular_values_all = np.sqrt(eigvals)
        nonzero = singular_values_all > 1.0e-12
        singular_values_all = singular_values_all[nonzero]
        left = left[:, nonzero]
        basis_all = np.dot(centered.T, left) / singular_values_all
        basis_all = basis_all.T
    else:
        cov = np.dot(centered.T, centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        singular_values_all = np.sqrt(eigvals)
        nonzero = singular_values_all > 1.0e-12
        singular_values_all = singular_values_all[nonzero]
        basis_all = eigvecs[:, order][:, nonzero].T

    if singular_values_all.size == 0:
        raise RuntimeError("All singular values are numerically zero")

    total_energy = float(np.sum(singular_values_all ** 2))
    explained = (singular_values_all ** 2) / total_energy
    cumulative = np.cumsum(explained)

    if n_components and n_components > 0:
        k = int(n_components)
    else:
        k = int(np.searchsorted(cumulative, variance) + 1)
        k = min(k, int(max_components))
    k = max(1, min(k, singular_values_all.size))

    basis = basis_all[:k, :]
    singular_values = singular_values_all[:k]
    coefficients = np.dot(centered, basis.T)
    return mean, basis, singular_values, explained[:k], cumulative[k - 1], coefficients


def relative_rmse(y_true, y_pred, center):
    numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.sqrt(np.mean((y_true - center) ** 2))
    if denom <= 1.0e-12:
        return float("nan")
    return float(numerator / denom)


def summarize_errors(Y, Y_hat, center):
    rmse = float(np.sqrt(np.mean((Y - Y_hat) ** 2)))
    rel = relative_rmse(Y, Y_hat, center)
    mae = float(np.mean(np.abs(Y - Y_hat)))
    return {"rmse": rmse, "relative_rmse": rel, "mae": mae}


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def train_surrogate(args):
    campaign_dir = Path(args.campaign_dir).resolve()
    config = load_json(campaign_dir / "sim_bo_config.json")
    records, image_shape, collection_summary = collect_records(
        campaign_dir,
        field=args.field,
        normalization=args.normalization,
        include_holdout=args.include_holdout,
    )

    if args.mode == "candidate-grid":
        X, Y, groups, input_names, dataset_summary = build_candidate_grid_dataset(
            records, config, args.condition_set
        )
    elif args.mode == "condition-field":
        X, Y, groups, input_names, dataset_summary = build_condition_field_dataset(records)
    else:
        raise ValueError("Unknown mode: {0}".format(args.mode))

    train_idx, test_idx = group_split(groups, args.test_fraction, args.random_seed)
    if train_idx.size == 0:
        raise RuntimeError("Empty training split")

    X_scaled_all, input_mean, input_scale = standardize_inputs(X[train_idx], X)
    regressor = args.regressor
    if regressor == "auto":
        if train_idx.size <= args.max_kernel_samples:
            regressor = "rbf-krr"
        else:
            regressor = "poly-ridge"

    powers = monomial_powers(X.shape[1], args.degree)
    Phi_all = polynomial_features(X_scaled_all, powers)

    field_mean, basis, singular_values, explained, retained_energy, coeff_train = compute_pod(
        Y[train_idx],
        n_components=args.n_components,
        variance=args.variance,
        max_components=args.max_components,
    )
    coeff_all = np.dot(Y - field_mean, basis.T)
    coeff_mean = np.mean(coeff_train, axis=0)
    coeff_scale = np.std(coeff_train, axis=0)
    coeff_scale[coeff_scale <= 1.0e-12] = 1.0
    coeff_scaled_all = (coeff_all - coeff_mean) / coeff_scale

    weights = np.asarray([], dtype=np.float64)
    dual_weights = np.asarray([], dtype=np.float64)
    rbf_length_scale = 0.0
    train_inputs_scaled = X_scaled_all[train_idx]

    if regressor == "poly-ridge":
        weights = fit_ridge(
            Phi_all[train_idx],
            coeff_scaled_all[train_idx],
            alpha=args.ridge_alpha,
        )
        coeff_pred_scaled = np.dot(Phi_all, weights)
    elif regressor == "rbf-krr":
        dual_weights, rbf_length_scale = fit_rbf_krr(
            train_inputs_scaled,
            coeff_scaled_all[train_idx],
            alpha=args.ridge_alpha,
            length_scale=args.rbf_length_scale,
        )
        K_all_train = rbf_kernel(X_scaled_all, train_inputs_scaled, rbf_length_scale)
        coeff_pred_scaled = np.dot(K_all_train, dual_weights)
    else:
        raise ValueError("Unknown regressor: {0}".format(regressor))

    coeff_pred = coeff_pred_scaled * coeff_scale + coeff_mean
    Y_pred = field_mean + np.dot(coeff_pred, basis)

    train_errors = summarize_errors(Y[train_idx], Y_pred[train_idx], field_mean)
    test_errors = None
    if test_idx.size:
        test_errors = summarize_errors(Y[test_idx], Y_pred[test_idx], field_mean)

    output_dir = Path(args.output_dir) if args.output_dir else campaign_dir / "svd_surrogate"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "{0}_model.npz".format(args.mode.replace("-", "_"))
    summary_path = output_dir / "{0}_summary.json".format(args.mode.replace("-", "_"))

    summary = {
        "campaign_dir": str(campaign_dir),
        "mode": args.mode,
        "field": args.field,
        "normalization": args.normalization,
        "include_holdout": bool(args.include_holdout),
        "condition_set": args.condition_set if args.mode == "candidate-grid" else None,
        "image_shape": list(image_shape),
        "input_names": input_names,
        "sample_count": int(X.shape[0]),
        "target_dimension": int(Y.shape[1]),
        "train_sample_count": int(train_idx.size),
        "test_sample_count": int(test_idx.size),
        "group_count": int(len(set(groups.tolist()))),
        "polynomial_degree": int(args.degree),
        "polynomial_feature_count": int(len(powers)),
        "regressor": regressor,
        "ridge_alpha": float(args.ridge_alpha),
        "rbf_length_scale": float(rbf_length_scale),
        "component_count": int(basis.shape[0]),
        "retained_energy": float(retained_energy),
        "singular_values": singular_values.tolist(),
        "explained_variance": explained.tolist(),
        "train_errors": train_errors,
        "test_errors": test_errors,
        "dataset": dataset_summary,
        "collection": collection_summary,
        "model_path": str(model_path),
    }
    summary = to_jsonable(summary)

    np.savez_compressed(
        str(model_path),
        mode=np.asarray([args.mode]),
        regressor=np.asarray([regressor]),
        field=np.asarray([args.field]),
        normalization=np.asarray([args.normalization]),
        image_shape=np.asarray(image_shape, dtype=np.int64),
        input_names=np.asarray(input_names),
        input_mean=input_mean,
        input_scale=input_scale,
        feature_powers=np.asarray(powers, dtype=np.int64),
        ridge_alpha=np.asarray([float(args.ridge_alpha)]),
        rbf_length_scale=np.asarray([float(rbf_length_scale)]),
        weights=weights,
        dual_weights=dual_weights,
        train_inputs_scaled=train_inputs_scaled,
        coeff_mean=coeff_mean,
        coeff_scale=coeff_scale,
        field_mean=field_mean,
        basis=basis,
        singular_values=singular_values,
        explained_variance=explained,
        X=X,
        groups=groups,
        train_indices=train_idx,
        test_indices=test_idx,
        condition_keys=np.asarray(dataset_summary.get("condition_keys", [])),
        summary_json=np.asarray([json.dumps(summary, sort_keys=True)]),
    )
    write_json(summary_path, summary)
    return summary


def load_model(path):
    data = np.load(str(path), allow_pickle=False)
    model = {key: data[key] for key in data.files}
    return model


def predict_coefficients(model, X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_scaled = (X - model["input_mean"]) / model["input_scale"]
    regressor = str(model["regressor"][0])
    if regressor == "rbf-krr":
        K = rbf_kernel(
            X_scaled,
            model["train_inputs_scaled"],
            float(model["rbf_length_scale"][0]),
        )
        coeff_scaled = np.dot(K, model["dual_weights"])
    else:
        Phi = polynomial_features(X_scaled, [tuple(row) for row in model["feature_powers"]])
        coeff_scaled = np.dot(Phi, model["weights"])
    return coeff_scaled * model["coeff_scale"] + model["coeff_mean"]


def reconstruct_from_coefficients(model, coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    return model["field_mean"] + np.dot(coefficients, model["basis"])


def main():
    default_campaign = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train an SVD/POD surrogate for simulated surface uz/disp_z."
    )
    parser.add_argument("--campaign-dir", default=str(default_campaign))
    parser.add_argument(
        "--mode",
        choices=("candidate-grid", "condition-field"),
        default="candidate-grid",
    )
    parser.add_argument(
        "--field",
        default="disp_z",
        help="NPZ field to use; use 'auto' to pick disp_z/uz/polar_z in order.",
    )
    parser.add_argument(
        "--normalization",
        choices=("znorm", "center", "raw"),
        default="znorm",
        help="Per-surface preprocessing before SVD.",
    )
    parser.add_argument(
        "--condition-set",
        choices=("common", "train", "all"),
        default="common",
        help="Condition selection for candidate-grid mode.",
    )
    parser.add_argument("--exclude-holdout", dest="include_holdout", action="store_false")
    parser.set_defaults(include_holdout=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument(
        "--regressor",
        choices=("auto", "poly-ridge", "rbf-krr"),
        default="auto",
        help="Coefficient regressor. auto uses RBF KRR when the train set is small.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--rbf-length-scale",
        type=float,
        default=0.0,
        help="RBF length scale in standardized input units. 0 uses median distance.",
    )
    parser.add_argument("--max-kernel-samples", type=int, default=2000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument(
        "--n-components",
        type=int,
        default=0,
        help="Number of SVD components. 0 chooses from --variance and --max-components.",
    )
    parser.add_argument("--variance", type=float, default=0.999)
    parser.add_argument("--max-components", type=int, default=64)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    summary = train_surrogate(args)
    print("Trained {0} surrogate".format(summary["mode"]))
    print("  samples: {0} train / {1} test".format(
        summary["train_sample_count"], summary["test_sample_count"]
    ))
    print("  components: {0} retained_energy={1:.6f}".format(
        summary["component_count"], summary["retained_energy"]
    ))
    print("  train relative_rmse: {0:.6f}".format(
        summary["train_errors"]["relative_rmse"]
    ))
    if summary["test_errors"] is not None:
        print("  test relative_rmse: {0:.6f}".format(
            summary["test_errors"]["relative_rmse"]
        ))
    print("  model: {0}".format(summary["model_path"]))


if __name__ == "__main__":
    main()
