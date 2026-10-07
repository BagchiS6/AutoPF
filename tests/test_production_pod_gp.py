from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autopf.closed_loop import Observation
from autopf.closed_loop.async_controller import incorporate_and_propose
from autopf.closed_loop.objectives import ObjectiveEvaluator, ObjectiveMode
from autopf.closed_loop.pod_gp import PODGaussianProcess
from autopf.closed_loop.production import (
    CandidateLibrary,
    ProductionPODGPStrategy,
    resolve_objective_mode,
)


class ObjectiveTests(unittest.TestCase):
    def test_pixel_and_posterior_objectives_rank_better_field_first(self):
        observed = np.asarray([[0.0, 1.0], [2.0, 3.0]])
        predictions = np.asarray([observed + 0.05, [[3.0, -1.0], [0.5, 2.0]]])
        pixel = ObjectiveEvaluator(ObjectiveMode.PIXEL_NMSE).score(observed, predictions)
        self.assertLess(pixel.scores[0], pixel.scores[1])

        basis = np.eye(4)[:2]
        covariance = np.eye(2) * 0.1
        posterior = ObjectiveEvaluator(
            ObjectiveMode.POSTERIOR_DISTANCE,
            feature_basis=basis,
            experimental_covariance=covariance,
        ).score(observed, predictions, np.ones_like(predictions) * 0.01)
        self.assertLess(posterior.scores[0], posterior.scores[1])
        self.assertEqual(posterior.diagnostics["feature_dimension"], 2)

    def test_objective_policy_alias_receipt_and_explicit_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "agent_choice.json"
            receipt.write_text(json.dumps({"objective_mode": "mahalanobis"}))
            self.assertIs(
                resolve_objective_mode(None, agent_choice_path=receipt),
                ObjectiveMode.POSTERIOR_DISTANCE,
            )
            self.assertIs(
                resolve_objective_mode("nmse", agent_choice_path=receipt),
                ObjectiveMode.PIXEL_NMSE,
            )
            receipt.write_text(json.dumps({"choice": "posterior_distance"}))
            with self.assertRaisesRegex(ValueError, "objective_mode"):
                resolve_objective_mode(None, agent_choice_path=receipt)

    def test_objective_rejects_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "experimental_covariance"):
            ObjectiveEvaluator(
                ObjectiveMode.POSTERIOR_DISTANCE,
                feature_basis=np.ones((2, 4)),
                experimental_covariance=np.eye(3),
            )
        evaluator = ObjectiveEvaluator(ObjectiveMode.PIXEL_NMSE, mask=np.ones(3))
        with self.assertRaisesRegex(ValueError, "mask"):
            evaluator.score(np.ones((2, 2)), np.ones((1, 2, 2)))


class CandidateLibraryTests(unittest.TestCase):
    def test_rejects_empty_duplicate_and_ragged_candidates(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            CandidateLibrary([])
        with self.assertRaisesRegex(ValueError, "unique"):
            CandidateLibrary([
                {"candidate_id": "same", "features": [0.0]},
                {"candidate_id": "same", "features": [1.0]},
            ])
        with self.assertRaisesRegex(ValueError, "same non-empty"):
            CandidateLibrary([
                {"candidate_id": "a", "features": [0.0]},
                {"candidate_id": "b", "features": [1.0, 2.0]},
            ])


class PODGPTests(unittest.TestCase):
    def test_fit_predict_save_and_online_condition(self):
        x = np.linspace(-1, 1, 8)[:, None]
        fields = np.asarray([
            [[value, value**2], [-value, 0.5 * value]] for value in x[:, 0]
        ])
        model = PODGaussianProcess(max_modes=4).fit(x, fields)
        mean, variance = model.predict(x)
        self.assertEqual(mean.shape, fields.shape)
        self.assertLess(float(np.mean((mean - fields) ** 2)), 0.02)
        self.assertTrue(np.all(variance >= 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = PODGaussianProcess.load(path)
            loaded_mean, loaded_variance = loaded.predict(x)
            np.testing.assert_allclose(loaded_mean, mean)
            np.testing.assert_allclose(loaded_variance, variance)
            frozen_basis = loaded._state["basis"].copy()
            frozen_length = float(loaded._state["length_scale"])
            loaded.update(np.asarray([[1.25]]), np.asarray([[[1.25, 1.25**2], [-1.25, 0.625]]]))
            self.assertEqual(loaded.training_count, 9)
            np.testing.assert_array_equal(loaded._state["basis"], frozen_basis)
            self.assertEqual(float(loaded._state["length_scale"]), frozen_length)
            self.assertTrue(loaded.receipt()["pod_basis_frozen_after_initial_fit"])

    def test_unfitted_and_one_sample_models_fail_loudly(self):
        model = PODGaussianProcess()
        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            model.predict(np.zeros((1, 1)))
        with self.assertRaisesRegex(ValueError, "At least two"):
            model.fit(np.zeros((1, 1)), np.zeros((1, 2, 2)))


class AsyncControllerTests(unittest.TestCase):
    def test_each_completion_updates_model_and_replenishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = CandidateLibrary([
                {
                    "candidate_id": f"c{index}",
                    "features": [float(index) / 5.0],
                    "parameters": {"Materials/hidden/value": float(index) / 5.0},
                }
                for index in range(6)
            ])
            strategy = ProductionPODGPStrategy(
                candidate_library=candidates,
                acquisition_plan=[{"voltage_v": 5.0, "pulse_s": 0.3}],
                model_path=root / "model.npz",
                objective=ObjectiveEvaluator(ObjectiveMode.PIXEL_NMSE),
                condition_features=["voltage_v", "pulse_s"],
                batch_size=1,
                bootstrap_size=2,
                async_evaluations_per_observation=4,
                online_release_minimum=2,
                async_context_root=root / "contexts",
            )
            observation = Observation(
                observation_id="obs-0",
                request_id="req-0",
                iteration=0,
                data={"surface_uz": [[0.0, 0.2], [0.4, 0.6]]},
                data_references={"pfm_path": "/instrument/frame-000.npy"},
                metadata={"condition": {"voltage_v": 5.0, "pulse_s": 0.3}},
            )
            strategy.observation_parameter_map = {
                "data_references.pfm_path": "UserObjects/exp/file"
            }
            requests = list(strategy.propose_simulations(
                {"strategy_state": {}}, observation
            ))
            self.assertEqual(len(requests), 2)
            self.assertTrue(all(
                row.parameters["UserObjects/exp/file"] == "/instrument/frame-000.npy"
                for row in requests
            ))
            context = requests[0].metadata["async_context_path"]

            queue = list(requests)
            completed = 0
            while queue:
                request = queue.pop(0)
                value = float(request.metadata["model_input"][0])
                field = [[value, value + 0.1], [2 * value, 2 * value + 0.1]]
                proposed = incorporate_and_propose(
                    {
                        "simulation_id": request.simulation_id,
                        "iteration": request.iteration,
                        "status": "completed",
                        "outputs": {"surface_uz": field},
                    },
                    context,
                )
                completed += 1
                if proposed is not None:
                    from autopf.closed_loop import SimulationRequest

                    proposed_request = SimulationRequest.from_dict(proposed)
                    self.assertEqual(
                        proposed_request.parameters["UserObjects/exp/file"],
                        "/instrument/frame-000.npy",
                    )
                    queue.append(proposed_request)
            state = json.loads(Path(json.loads(Path(context).read_text())["state_path"]).read_text())
            self.assertEqual(completed, 4)
            self.assertEqual(state["completed"], 4)
            self.assertFalse(state["pending"])
            self.assertEqual(len(state["launched_pairs"]), len(set(state["launched_pairs"])))
            self.assertEqual(PODGaussianProcess.load(root / "model.npz").training_count, 4)

            first = state["completed_results"][0]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                incorporate_and_propose(first, context)


if __name__ == "__main__":
    unittest.main()
