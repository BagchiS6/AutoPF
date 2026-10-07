"""Site-owned scientific hooks required by the production launcher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def calibrated_pfm_to_uz(array, result):
    """Convert the Tiled PFM channel to calibrated displacement.

    Replace this placeholder with the reviewed amplitude/phase calibration.
    Returning raw amplitude as displacement is intentionally forbidden.
    """

    raise NotImplementedError("Install the microscope-specific PFM-to-u_z calibration.")


def load_moose_surface_uz(request, run_directory: Path):
    """Return a full MOOSE surface field from one completed run directory.

    This reference expects a postprocessor to write ``surface_uz.npy``.  Change
    only this hook when the MOOSE application emits Exodus, CSV, or another
    reviewed format.
    """

    field_path = Path(run_directory) / "surface_uz.npy"
    if not field_path.is_file():
        raise FileNotFoundError(f"MOOSE run did not produce {field_path}")
    field = np.load(field_path, allow_pickle=False)
    receipt = {
        "candidate_id": request.metadata["candidate_id"],
        "shape": list(field.shape),
        "source": str(field_path),
    }
    (Path(run_directory) / "autopf_field_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "completed",
        "data_references": {"surface_uz": str(field_path)},
        "candidate_id": request.metadata["candidate_id"],
    }


def build_moose_command(request, work_directory, config):
    """Build one MatEnsemble chore for the reviewed site task wrapper.

    The wrapper must run the configured MOOSE application, extract the full
    surface field, and print one final JSON object with ``simulation_id``,
    ``status``, and ``data_references.surface_uz``.  It runs as a 64-rank MPI
    chore; application-specific launch and rank-zero extraction remain a site
    responsibility rather than being guessed by AutoPF.
    """

    values = dict(request.parameters)
    mapping = dict(config["condition_override_map"])
    for key, value in request.condition.items():
        values[mapping[key]] = value
    arguments = [f"{key}={value}" for key, value in sorted(values.items())]
    return [
        str(config["task_wrapper"]),
        "--simulation-id", request.simulation_id,
        "--iteration", str(request.iteration),
        "--workdir", str(work_directory),
        "--moose-app", str(config["moose_app"]),
        "--input", str(config["base_input"]),
        "--metadata-json", json.dumps(request.metadata, separators=(",", ":")),
        "--",
        *arguments,
    ]
