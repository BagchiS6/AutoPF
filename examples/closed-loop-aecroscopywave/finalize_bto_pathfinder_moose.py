#!/usr/bin/env python3
"""Extract a normalized 52x52 top-surface displacement from one BTO solve."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np


def load_campaign_utils(path: Path):
    spec = importlib.util.spec_from_file_location("autopf_campaign_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load campaign utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finalize(args) -> dict:
    if args.moose_returncode != 0:
        raise RuntimeError(f"MOOSE application returned {args.moose_returncode}")
    utilities = load_campaign_utils(Path(args.campaign_utils))
    from surface_orientation import apply_surface_orientation, resample_array

    run_dir = args.run_dir.resolve()
    exodus = utilities.newest_file(run_dir, "*.e")
    if exodus is None:
        raise RuntimeError(f"No Exodus output found in {run_dir}")
    parameters = json.loads(args.parameters_json)
    run_spec = {
        "output_dir": str(run_dir),
        "postprocess_script": str(Path(args.postprocess_script).resolve()),
        "python": args.python,
        "tip_voltage": float(parameters.get("tip_voltage", parameters.get("voltage_v", 0.0))),
        "parameters": parameters,
    }
    postprocessed = utilities.postprocess_surface_fields_from_result({
        "status": "completed",
        "returncode": 0,
        "run_spec": run_spec,
        "output_dir": str(run_dir),
        "exodus_file": str(exodus),
        "command": [],
    })
    if postprocessed.get("status") not in {"postprocessed", "reused_postprocessed"}:
        raise RuntimeError(f"Surface postprocessing failed: {postprocessed}")
    with np.load(postprocessed["fields_path"], allow_pickle=False) as archive:
        field = np.asarray(archive["disp_z"] if "disp_z" in archive else archive["uz"], dtype=float)
    field = utilities.crop_simulation_to_physical_fov(field, 350.0, 500.0)
    field = apply_surface_orientation(field, "identity")
    field = resample_array(field, (64, 64))
    field = utilities.znorm(utilities.center_crop(field, 0.1))
    field_path = run_dir / "surface_uz.npy"
    np.save(field_path, field, allow_pickle=False)
    return {
        "simulation_id": args.simulation_id,
        "iteration": args.iteration,
        "status": "completed",
        "outputs": {},
        "data_references": {
            "surface_uz": str(field_path),
            "raw_surface_fields": str(postprocessed["fields_path"]),
            "exodus": str(exodus),
        },
        "metadata": {
            **json.loads(args.metadata_json),
            "walltime_s": time.time() - args.start_epoch,
            "surface_processing": "350/500 FOV crop; identity orientation; 64x64 resample; 10% crop; z-normalize",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--simulation-id", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--start-epoch", type=float, required=True)
    parser.add_argument("--moose-returncode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pickle-output", type=Path, required=True)
    parser.add_argument("--campaign-utils", default=str(Path(
        "/scratch/hpcl-mat269/sny/sim_bo_bto_physical_bounds/agent_handoff_20260828/"
        "autopf_migration_staging/autopf/utils.py"
    )))
    parser.add_argument("--postprocess-script", default=str(Path(
        "/scratch/hpcl-mat269/sny/sim_bo_bto_physical_bounds/agent_handoff_20260828/"
        "autopf_migration_staging/campaign_assets/plot_exodus_fields.py"
    )))
    parser.add_argument("--python", default="/opt/basic/bin/python")
    args = parser.parse_args()
    try:
        result = finalize(args)
    except Exception as exc:
        result = {
            "simulation_id": args.simulation_id,
            "iteration": args.iteration,
            "status": "failed",
            "outputs": {},
            "data_references": {},
            "metadata": {
                **json.loads(args.metadata_json),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.pickle_output.open("wb") as stream:
        pickle.dump(result, stream)
    print(json.dumps(result), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
