from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from autopf.closed_loop import JsonStateStore, SimulationRequest
from autopf.closed_loop.backends import (
    AEcroscopyWaveAcquisitionBackend,
    MatEnsembleAsyncTSBackend,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def load_file_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


RUNNER = load_file_module(
    "autopf_test_run_production",
    REPOSITORY / "examples/closed-loop-aecroscopywave/run_production.py",
)
BTO_HOOKS = load_file_module(
    "autopf_test_bto_hooks",
    REPOSITORY / "examples/closed-loop-aecroscopywave/bto_pathfinder_hooks.py",
)
GENERIC_BTO_HOOKS = load_file_module(
    "autopf_test_generic_bto_hooks",
    REPOSITORY / "examples/closed-loop-aecroscopywave/bto_moose_hooks.py",
)


class ProductionWiringTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        scratch = root / "scratch"
        scratch.mkdir(parents=True)
        application = root / "bto-opt"
        application.write_text("fake executable used only for preflight\n")
        application.chmod(0o755)
        base_input = root / "base.i"
        base_input.write_text("[]\n")
        candidates = root / "candidates.json"
        candidates.write_text(json.dumps({"candidates": [
            {"candidate_id": "hidden-off", "features": [0.0], "parameters": {"hidden": 0.0}},
            {"candidate_id": "vegard", "features": [1.0], "parameters": {"hidden": 1.0}},
        ]}))
        plan = root / "plan.json"
        plan.write_text(json.dumps({"acquisitions": [
            {"job_type": "pfm_image", "voltage_v": 5.0, "pulse_s": 0.3}
        ]}))
        return {
            "campaign": {
                "campaign_id": "contract-test",
                "state_path": str(scratch / "campaign.json"),
            },
            "acquisition": {
                "server_url": "https://microscope.invalid",
                "instrument_id": "pfm-test",
                "secret_env": "AUTOPF_TEST_SECRET",
                "pfm_transform": "autopf_contract_hooks:transform",
                "tiled_array_key": "pfm",
            },
            "simulation": {
                "backend": "async_ts",
                "scratch_root": str(scratch),
                "output_root": str(scratch / "moose-runs"),
                "moose_app": str(application),
                "base_input": str(base_input),
                "command_builder": "autopf_contract_hooks:build_command",
                "command_config": {"task_wrapper": "/fake/wrapper"},
                "num_cores": 4,
                "condition_override_map": {
                    "voltage_v": "BCs/tip/potential",
                    "pulse_s": "Executioner/end_time",
                },
            },
            "strategy": {
                "objective_mode": "pixel_nmse",
                "candidate_library": str(candidates),
                "acquisition_plan": str(plan),
                "model_path": str(scratch / "pod_gp.npz"),
                "condition_features": ["voltage_v", "pulse_s"],
                "batch_size": 2,
                "bootstrap_size": 2,
                "online_release_minimum": 2,
                "async_evaluations_per_observation": 4,
            },
        }

    def test_launcher_builds_only_live_acquisition_and_real_moose_backends(self):
        hooks = types.ModuleType("autopf_contract_hooks")
        hooks.transform = lambda array, _result: array
        hooks.build_command = lambda request, workdir, config: ["true"]
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            with patch.dict(sys.modules, {"autopf_contract_hooks": hooks}), patch.dict(
                "os.environ", {"AUTOPF_TEST_SECRET": "test-only-secret"}, clear=False
            ):
                store, acquisition, simulation, strategy, metadata = RUNNER.production_components(
                    config, None
                )
            self.assertIsInstance(acquisition, AEcroscopyWaveAcquisitionBackend)
            self.assertIsInstance(simulation, MatEnsembleAsyncTSBackend)
            self.assertEqual(strategy.objective.mode.value, "pixel_nmse")
            self.assertFalse(metadata["proxy_acquisition"])
            self.assertFalse(metadata["proxy_simulation"])
            self.assertEqual(store.path, Path(config["campaign"]["state_path"]).resolve())

    def test_site_environment_expansion_is_recursive(self):
        with patch.dict("os.environ", {"AUTOPF_SITE_ROOT": "/pscratch/test/autopf"}):
            expanded = RUNNER.expand_environment({
                "path": "$AUTOPF_SITE_ROOT/state.json",
                "nested": ["$AUTOPF_SITE_ROOT/model.npz"],
            })
        self.assertEqual(expanded["path"], "/pscratch/test/autopf/state.json")
        self.assertEqual(expanded["nested"], ["/pscratch/test/autopf/model.npz"])

    def test_launcher_enforces_scratch_and_frozen_objective(self):
        hooks = types.ModuleType("autopf_contract_hooks")
        hooks.transform = lambda array, _result: array
        hooks.build_command = lambda request, workdir, config: ["true"]
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            config["simulation"]["output_root"] = str(Path(directory) / "outside")
            with patch.dict(sys.modules, {"autopf_contract_hooks": hooks}):
                with self.assertRaisesRegex(ValueError, "inside scratch_root"):
                    RUNNER.production_components(config, None, allow_offline_instrument=True)

            config = self._fixture(Path(directory) / "second")
            JsonStateStore(config["campaign"]["state_path"]).initialize(
                "contract-test", {"objective_mode": "pixel_nmse"}
            )
            artifact = Path(directory) / "objective.npz"
            np.savez(artifact, basis=np.eye(4)[:2], covariance=np.eye(2))
            config["strategy"]["objective_artifact"] = str(artifact)
            with patch.dict(sys.modules, {"autopf_contract_hooks": hooks}):
                with self.assertRaisesRegex(ValueError, "froze objective_mode"):
                    RUNNER.production_components(
                        config, "posterior_distance", allow_offline_instrument=True
                    )


class BTOCommandContractTests(unittest.TestCase):
    def test_command_contains_identity_metadata_and_mapped_moose_overrides(self):
        request = SimulationRequest(
            simulation_id="moose-00000-vegard",
            iteration=3,
            parameters={"Materials/vegard/eigenstrain": 0.012},
            condition={"voltage_v": -4.0, "pulse_s": 0.2},
            metadata={"candidate_id": "vegard", "objective_mode": "posterior_distance"},
        )
        command = BTO_HOOKS.build_moose_command(
            request,
            "/scratch/workdir",
            {
                "task_wrapper": "/opt/run_moose.sh",
                "moose_app": "/opt/bto-opt",
                "base_input": "/opt/bto.i",
                "condition_override_map": {
                    "voltage_v": "BCs/tip/potential",
                    "pulse_s": "Executioner/end_time",
                },
            },
        )
        self.assertEqual(command[:4], [
            "/opt/run_moose.sh", "/scratch/workdir", "moose-00000-vegard", "3"
        ])
        self.assertEqual(command[6:9], ["/opt/bto-opt", "-i", "/opt/bto.i"])
        self.assertIn("BCs/tip/potential=-4.0", command)
        self.assertIn("Executioner/end_time=0.2", command)
        metadata = json.loads(command[4])
        values = json.loads(command[5])
        self.assertEqual(metadata["candidate_id"], "vegard")
        self.assertEqual(values["Materials/vegard/eigenstrain"], 0.012)

    def test_site_neutral_hook_rejects_unmapped_conditions(self):
        request = SimulationRequest(
            simulation_id="unmapped",
            iteration=0,
            parameters={},
            condition={"voltage_v": 4.0},
        )
        with self.assertRaisesRegex(KeyError, "voltage_v"):
            GENERIC_BTO_HOOKS.build_moose_command(
                request,
                "/scratch/workdir",
                {
                    "task_wrapper": "/opt/run_moose.sh",
                    "moose_app": "/opt/application-opt",
                    "base_input": "/opt/model.i",
                    "condition_override_map": {},
                },
            )


class NERSCProfileTests(unittest.TestCase):
    def test_perlmutter_profile_has_required_account_resources_and_pins(self):
        directory = REPOSITORY / "examples/closed-loop-aecroscopywave"
        batch = (directory / "submit_nersc_perlmutter_production.slurm").read_text()
        self.assertIn("#SBATCH --account=m5014_g", batch)
        self.assertIn("#SBATCH --constraint=gpu", batch)
        self.assertIn("#SBATCH --qos=regular", batch)
        self.assertIn("#SBATCH --gpus-per-node=4", batch)
        self.assertIn("dd3d84a665", batch)
        self.assertIn("c7809ccee6216ece36b7767c30ba0b2f5cb4f4f6", batch)

        config = json.loads(
            (directory / "production_config.nersc-perlmutter.template.json").read_text()
        )
        self.assertEqual(config["simulation"]["scratch_root"], "$SCRATCH")
        self.assertEqual(config["simulation"]["backend"], "async_ts")
        self.assertEqual(
            config["simulation"]["command_builder"],
            "bto_moose_hooks:build_moose_command",
        )
        self.assertEqual(config["strategy"]["objective_mode"], "posterior_distance")


if __name__ == "__main__":
    unittest.main()
