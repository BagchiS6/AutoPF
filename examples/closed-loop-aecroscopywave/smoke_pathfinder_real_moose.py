#!/usr/bin/env python3
"""Release gate: four real BTO MOOSE solves with per-completion POD--GP TS."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from autopf.closed_loop import Observation
from autopf.closed_loop.backends import MatEnsembleAsyncTSBackend
from autopf.closed_loop.objectives import ObjectiveEvaluator, ObjectiveMode
from autopf.closed_loop.production import CandidateLibrary, ProductionPODGPStrategy


ROOT = Path("/scratch/hpcl-mat269/sny/sim_bo_bto_physical_bounds")
REPO = Path("/scratch/hpcl-mat269/sny/AutoPF_workflow_integration")
OUTPUT = Path(os.environ["AUTOPF_SMOKE_OUTPUT"]).resolve()

HIDDEN_NAMES = (
    "leg_screen_c0", "leg_screen_cx", "leg_screen_cy", "ring_screen_amp",
    "leg_anis_yz_c0", "leg_anis_yz_cx", "leg_anis_yz_cy", "odd_ring_anis_yz_amp",
    "leg_anis_xz_c0", "leg_anis_xz_cx", "leg_anis_xz_cy",
    "leg_flexo_c0", "leg_flexo_cx", "leg_flexo_cy", "leg_flexo_grad_c0",
)
ZERO_PHYSICS = {
    "screen_lambda": 0.0, "vegard_strain": 0.0,
    "spatial_screen_amp": 0.0, "spatial_vegard_amp": 0.0,
    "anis_vegard_xx_amp": 0.0, "anis_vegard_yy_amp": 0.0,
    "anis_vegard_zz_amp": 0.0, "anis_vegard_xy_amp": 0.0,
    "anis_vegard_xz_amp": 0.0, "anis_vegard_yz_amp": 0.0,
    "flexo_proxy_amp": 0.0, "flexo_proxy_grad_amp": 0.0,
    "g11": 0.5, "g12": -0.06, "g44": 0.02,
    "spatial_screen_sigma_xy": 80.0, "spatial_screen_sigma_z": 25.0,
    "spatial_screen_x0": 0.0, "spatial_screen_y0": 0.0, "spatial_screen_z0": 100.0,
    "spatial_vegard_sigma_xy": 120.0, "spatial_vegard_sigma_z": 35.0,
    "spatial_vegard_x0": 0.0, "spatial_vegard_y0": 0.0, "spatial_vegard_z0": 100.0,
    "anis_vegard_sigma_xy": 120.0, "anis_vegard_sigma_z": 35.0,
    "anis_vegard_x0": 0.0, "anis_vegard_y0": 0.0, "anis_vegard_z0": 100.0,
    "flexo_proxy_sigma_xy": 80.0, "flexo_proxy_sigma_z": 25.0,
    "flexo_proxy_x0": 0.0, "flexo_proxy_y0": 0.0, "flexo_proxy_z0": 100.0,
}


def candidate(candidate_id: str, **values: float) -> dict:
    hidden = {name: 0.0 for name in HIDDEN_NAMES}
    hidden.update(values)
    return {
        "candidate_id": candidate_id,
        "features": [hidden[name] for name in HIDDEN_NAMES],
        "parameters": {**ZERO_PHYSICS, **hidden},
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Release-gate output must be fresh: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    manifest = json.loads((ROOT / "pathfinder_setup/spatial_inversion_contract/calibration_manifest_v1.json").read_text())
    row = manifest["rows"][0]
    specs = list((
        ROOT / "pathfinder_runs/spatial_hidden_population_perlmutter_g_001/"
        "hidden_forward_multimodal_prospective_validation_v1/run_specs"
    ).glob("*.json"))
    spec = next(
        json.loads(path.read_text()) for path in specs
        if json.loads(path.read_text())["experiment_key"] == row["key"]
    )
    with np.load(row["npz_path"], allow_pickle=False) as archive:
        observed = np.asarray(archive["uz"], dtype=float)
    edge = int(round(0.1 * observed.shape[0]))
    observed = observed[edge:-edge, edge:-edge]
    observed = (observed - observed.mean()) / max(float(observed.std()), 1.0e-12)
    condition = {
        "tip_voltage": float(spec["tip_voltage"]),
        "pulse_end": float(spec["pulse_end"]),
        **{key: float(value) for key, value in spec["input_params"].items()},
    }
    library = CandidateLibrary([
        candidate("off"),
        candidate("screen", leg_screen_c0=0.04, ring_screen_amp=0.06),
        candidate("yz_pos", leg_anis_yz_c0=0.0005),
        candidate("screen_yz", leg_screen_c0=0.04, ring_screen_amp=0.06, leg_anis_yz_c0=0.0005),
    ])
    strategy = ProductionPODGPStrategy(
        candidate_library=library,
        acquisition_plan=[condition],
        model_path=OUTPUT / "model/online_pod_gp.pt",
        objective=ObjectiveEvaluator(ObjectiveMode.PIXEL_NMSE),
        condition_features=list(condition),
        batch_size=1,
        bootstrap_size=2,
        async_evaluations_per_observation=4,
        online_release_minimum=2,
        async_context_root=OUTPUT / "contexts",
        observation_parameter_map={"data_references.pfm_initial_condition": "pfm_image_file"},
        max_modes=8,
        max_inducing=4,
        gp_training_steps=40,
        gp_online_steps=6,
        gp_device="cpu",
    )
    observation = Observation(
        observation_id="stored-release-gate-observation",
        request_id="stored-release-gate-request",
        iteration=0,
        data={"surface_uz": observed.tolist()},
        data_references={"pfm_initial_condition": row["polar_z_path"]},
        metadata={"condition": condition, "release_gate_only": True},
    )
    requests = strategy.propose_simulations({"strategy_state": {}}, observation)
    task_pythonpath = ":".join((
        str(REPO), str(ROOT / "pathfinder_setup"),
        "/scratch/hpcl-mat269/sny/vendor/MatEnsemble/src",
        "/scratch/hpcl-mat269/sny/python_envs/autopf-gp-torch260/lib/python3.12/site-packages",
    ))
    backend = MatEnsembleAsyncTSBackend(
        output_root=OUTPUT / "moose",
        command_builder="bto_pathfinder_hooks:build_moose_command",
        command_config={
            "task_wrapper": str(REPO / "examples/closed-loop-aecroscopywave/run_bto_pathfinder_moose.sh"),
            "moose_app": "/opt/ferret/ferret-opt",
            "base_input": str(ROOT / "agent_handoff_20260828/autopf_migration_staging/"
                              "campaign_assets/BTO_DW_anisotropic_hidden_physics_zdecay_ic.i"),
            "condition_override_map": {key: key for key in condition},
        },
        num_cores=64,
        task_environment={
            "PYTHONPATH": task_pythonpath,
            "OMP_NUM_THREADS": "1",
            "MPICH_GPU_SUPPORT_ENABLED": "0",
            "OMPI_MCA_pml": "ob1",
            "OMPI_MCA_btl": "self,vader",
            "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
            "AUTOPF_PYTHON": "/opt/basic/bin/python",
            "AUTOPF_BTO_FINALIZER": str(
                REPO / "examples/closed-loop-aecroscopywave/finalize_bto_pathfinder_moose.py"
            ),
        },
        buffer_time=0.0,
        log_delay=2.0,
    )
    handle = backend.submit(requests)
    results = backend.collect(handle)
    if len(results) != 4 or any(result.status != "completed" for result in results):
        raise RuntimeError(f"Real-MOOSE release gate failed: {[result.to_dict() for result in results]}")
    receipt = {
        "schema": "autopf.pathfinder_real_moose_async_ts_smoke.v1",
        "passed": True,
        "real_moose_simulations": len(results),
        "per_completion_online_updates": len(results),
        "model_path": str(OUTPUT / "model/online_pod_gp.pt"),
        "batch_receipt": str(Path(handle.metadata["batch_directory"]) / "batch_results.json"),
        "proxy_simulation": False,
    }
    (OUTPUT / "release_gate_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
