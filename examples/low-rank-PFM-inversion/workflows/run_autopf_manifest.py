"""MatEnsemble workflow entry point for campaign-generated AutoPF manifests."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from autopf.utils import automoose, postproc

POSTPROCESS_WAIT_TIMEOUT_SECONDS = int(os.environ.get("AUTOPF_POSTPROCESS_WAIT_SECONDS", "300"))
POSTPROCESS_WAIT_INTERVAL_SECONDS = int(os.environ.get("AUTOPF_POSTPROCESS_WAIT_INTERVAL_SECONDS", "10"))
POSTPROCESS_STABLE_CHECK_SECONDS = int(os.environ.get("AUTOPF_POSTPROCESS_STABLE_CHECK_SECONDS", "5"))
POSTPROCESS_PATTERNS = (
    "fields_final_timestep.npz",
    "fields_final_timestep.png",
)


def _load_manifest() -> dict:
    manifest_path = Path(os.environ.get("AUTOPF_MANIFEST_PATH", "autopf_manifest.json"))
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.setdefault("manifest_path", str(manifest_path))
    return manifest


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _candidate_postprocessed_outputs(runs: list[dict]) -> list[tuple[dict, Path]]:
    candidates: list[tuple[dict, Path]] = []
    for run in runs:
        output_dir = Path(run["output_dir"])
        for pattern in POSTPROCESS_PATTERNS:
            for path in output_dir.rglob(pattern):
                candidates.append((run, path))
    return candidates


def _ready_postprocessed_outputs(runs: list[dict]) -> list[tuple[dict, Path]]:
    candidates = _candidate_postprocessed_outputs(runs)
    if not candidates:
        return []

    initial_sizes: dict[Path, int] = {}
    for _, path in candidates:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return []
        if size <= 0:
            return []
        initial_sizes[path] = size

    time.sleep(POSTPROCESS_STABLE_CHECK_SECONDS)

    for _, path in candidates:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return []
        if size <= 0 or size != initial_sizes[path]:
            return []

    return candidates


def _stage_postprocessed_outputs(campaign_dir: Path, ready_outputs: list[tuple[dict, Path]]) -> list[Path]:
    staged_root = campaign_dir / "postprocessed"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_files: list[Path] = []

    for run, path in ready_outputs:
        run_dir = staged_root / _safe_name(str(run.get("key") or Path(run["output_dir"]).name))
        run_dir.mkdir(parents=True, exist_ok=True)
        staged_path = run_dir / path.name
        shutil.copy2(path, staged_path)
        if staged_path.stat().st_size <= 0:
            raise RuntimeError(f"Refusing to transfer empty staged postprocessed file: {staged_path}")
        staged_files.append(staged_path)

    return staged_files


def _postprocessed_transfer_paths(campaign_dir: Path, ready_outputs: list[tuple[dict, Path]]) -> list[Path]:
    return _stage_postprocessed_outputs(campaign_dir, ready_outputs)


def _write_transfer_status(campaign_dir: Path, status: dict) -> None:
    status_path = campaign_dir / "result_transfer_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _wait_for_postprocessed_transfer_paths(campaign_dir: Path, runs: list[dict]) -> list[Path]:
    deadline = time.monotonic() + POSTPROCESS_WAIT_TIMEOUT_SECONDS
    attempt = 0
    while True:
        attempt += 1
        ready_outputs = _ready_postprocessed_outputs(runs)
        transfer_paths = _postprocessed_transfer_paths(campaign_dir, ready_outputs) if ready_outputs else []
        if transfer_paths:
            print(f"Found postprocessed outputs for result transfer staging on attempt {attempt}.")
            return transfer_paths

        run_dirs = [str(Path(run["output_dir"])) for run in runs]
        candidates = _candidate_postprocessed_outputs(runs)
        print(
            "Waiting for postprocessed field files before result transfer staging "
            f"(attempt {attempt}, checked {len(run_dirs)} run directories, "
            f"found {len(candidates)} candidate files)."
        )
        _write_transfer_status(
            campaign_dir,
            {
                "status": "waiting_for_postprocessed_files",
                "attempt": attempt,
                "run_dirs": run_dirs,
                "candidate_files": [
                    {"path": str(path), "size_bytes": path.stat().st_size if path.exists() else None}
                    for _, path in candidates
                ],
                "expected_files": list(POSTPROCESS_PATTERNS),
                "timeout_seconds": POSTPROCESS_WAIT_TIMEOUT_SECONDS,
                "stable_check_seconds": POSTPROCESS_STABLE_CHECK_SECONDS,
            },
        )

        if time.monotonic() >= deadline:
            return []
        time.sleep(POSTPROCESS_WAIT_INTERVAL_SECONDS)


def _stage_result_transfer(manifest: dict) -> None:
    config = manifest.get("result_transfer") or {}
    manifest_path = Path(manifest.get("manifest_path", "autopf_manifest.json"))
    campaign_dir = manifest_path.parent

    if config.get("method") != "rsync":
        print("Result transfer staging skipped: manifest did not request rsync.")
        _write_transfer_status(campaign_dir, {"status": "skipped", "config": config})
        return

    transfer_paths = _wait_for_postprocessed_transfer_paths(campaign_dir, manifest.get("runs") or [])
    if not transfer_paths:
        print("Result transfer staging skipped: no postprocessed field files were found.")
        _write_transfer_status(
            campaign_dir,
            {
                "status": "skipped_no_postprocessed_files",
                "expected_files": list(POSTPROCESS_PATTERNS),
                "timeout_seconds": POSTPROCESS_WAIT_TIMEOUT_SECONDS,
                "stable_check_seconds": POSTPROCESS_STABLE_CHECK_SECONDS,
            },
        )
        return

    transfer_list = campaign_dir / "result_transfer_files.txt"
    transfer_list.write_text(
        "\n".join(str(path) for path in transfer_paths) + "\n",
        encoding="utf-8",
    )
    rsync_manifest = campaign_dir / "rsync_result_transfer.json"
    rsync_manifest.write_text(
        json.dumps(
            {
                "remote_postprocessed_dir": str(campaign_dir / "postprocessed"),
                "local_results_dir": config.get("local_results_dir"),
                "files": [
                    {"path": str(path), "size_bytes": path.stat().st_size}
                    for path in transfer_paths
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_transfer_status(
        campaign_dir,
        {
            "status": "ready_for_rsync_pull",
            "method": "rsync",
            "transfer_list": str(transfer_list),
            "transfer_paths": [
                {"path": str(path), "size_bytes": path.stat().st_size}
                for path in transfer_paths
            ],
            "remote_postprocessed_dir": str(campaign_dir / "postprocessed"),
            "local_results_dir": config.get("local_results_dir"),
            "rsync_manifest": str(rsync_manifest),
        },
    )
    print(f"Postprocessed results are staged for rsync pull: {campaign_dir / 'postprocessed'}")


def run_manifest(manifest: dict) -> None:
    """Translate a campaign manifest into the AutoPF ``automoose`` call."""
    runs = manifest.get("runs") or []
    if not runs:
        raise ValueError("AutoPF manifest contains no runs")

    arg_list: list[list[str]] = []
    pp_arg_list: list[list[str]] = []
    directory_list: list[str] = []

    for run in runs:
        output_dir = Path(run["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        directory_list.append(str(output_dir))

        run_args = [
            f"tip_voltage={run['tip_voltage']}",
            f"pulse_end={run['pulse_end']}",
            f"g11={run['g11']}",
            f"g44={run['g44']}",
            f"pfm_image_file={run['pfm_image_file']}",
        ]
        if run.get("g12") is not None:
            run_args.insert(3, f"g12={run['g12']}")
        arg_list.append(run_args)
        pp_args = [
            f"--bias-voltage={run['tip_voltage']}",
            f"--g11={run['g11']}",
            *([f"--g12={run['g12']}"] if run.get("g12") is not None else []),
            f"--g44={run['g44']}",
            f"--tiled-path=phasefield/{run['key']}_surface_final",
        ]
        tiled_uri = os.environ.get("TILED_URI", "")
        tiled_api_key = os.environ.get("TILED_API_KEY", "")
        if tiled_uri:
            pp_args.append(f"--tiled-uri={tiled_uri}")
        if tiled_api_key:
            pp_args.append(f"--tiled-api-key={tiled_api_key}")
        pp_arg_list.append(pp_args)

    params = {
        "total_jobs": len(runs),
        "base_input": manifest["base_input"],
        "arg_list": arg_list,
        "num_cores": int(manifest.get("num_cores", 40)),
        "directory_list": directory_list,
    }

    print(
        f"Starting AutoPF campaign from {manifest.get('manifest_path')} "
        f"with {len(runs)} MatEnsemble jobs..."
    )
    automoose(manifest["executable"], params)
    print("AutoPF campaign completed. Starting Exodus post-processing...")
    postproc(
        ppscript=manifest.get("postprocess_script", "plot_exodus_fields.py"),
        params={
            "total_jobs": len(runs),
            "base_input": manifest["base_input"],
            "arg_list": pp_arg_list,
            "num_cores": 1,
            "directory_list": directory_list,
        },
    )
    _stage_result_transfer(manifest)


if __name__ == "__main__":
    run_manifest(_load_manifest())
