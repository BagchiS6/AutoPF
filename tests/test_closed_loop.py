from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autopf.closed_loop import (
    ClosedLoopRunner,
    ExperimentRequest,
    JsonStateStore,
    SimulationBatchHandle,
    SimulationRequest,
    SimulationResult,
    StrategyUpdate,
)
from autopf.closed_loop.backends import ProxyAcquisitionBackend, ProxySimulationBackend
from autopf.closed_loop.strategies import CallbackStrategy


def strategy():
    def experiment(_state, iteration):
        return ExperimentRequest(f"exp-{iteration}", iteration, payload={"x": iteration})

    def simulations(_state, observation):
        return [SimulationRequest(f"sim-{observation.iteration}", observation.iteration, {"x": observation.data["x"]})]

    def update(_state, observation, results):
        return StrategyUpdate(
            summary={"value": results[0].outputs["value"]},
            strategy_state={"last": observation.iteration},
        )

    return CallbackStrategy(
        propose_experiment=experiment,
        propose_simulations=simulations,
        update=update,
    )


class ClosedLoopTests(unittest.TestCase):
    def test_multiple_iterations_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            acquisition = ProxyAcquisitionBackend(lambda request: {"x": request.payload["x"]})
            simulation = ProxySimulationBackend(lambda request: {"value": request.parameters["x"] + 1})
            runner = ClosedLoopRunner(
                campaign_id="test",
                state_store=JsonStateStore(path),
                acquisition=acquisition,
                simulation=simulation,
                strategy=strategy(),
            )
            first = runner.run(2)
            self.assertEqual(first["next_iteration"], 2)
            # Continuing uses the same checkpoint and adds only new iterations.
            second = runner.run(4)
            self.assertEqual(second["next_iteration"], 4)
            self.assertEqual(len(second["iterations"]), 4)
            self.assertEqual(second["strategy_state"]["last"], 3)

    def test_submitted_acquisition_is_not_duplicated_after_collect_failure(self):
        class FailsOnce(ProxyAcquisitionBackend):
            def __init__(self):
                super().__init__(lambda request: {"x": request.payload["x"]})
                self.submits = 0
                self.fail = True

            def submit(self, request):
                self.submits += 1
                return super().submit(request)

            def collect(self, handle):
                if self.fail:
                    self.fail = False
                    raise RuntimeError("temporary acquisition outage")
                return super().collect(handle)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            acquisition = FailsOnce()
            runner = ClosedLoopRunner(
                campaign_id="resume",
                state_store=JsonStateStore(path),
                acquisition=acquisition,
                simulation=ProxySimulationBackend(lambda request: {"value": 1}),
                strategy=strategy(),
            )
            with self.assertRaisesRegex(RuntimeError, "temporary"):
                runner.run(1)
            checkpoint = json.loads(path.read_text())
            self.assertEqual(checkpoint["iterations"][0]["phase"], "acquisition_submitted")
            runner.run(1)
            self.assertEqual(acquisition.submits, 1)

    def test_adaptive_backend_may_return_replenishment_results(self):
        class AdaptiveSimulation:
            def submit(self, requests):
                return SimulationBatchHandle(
                    backend="adaptive-test",
                    handle_id="adaptive-0",
                    requests=tuple(requests),
                    metadata={"adaptive_requests": True},
                )

            def collect(self, handle):
                initial = handle.requests[0]
                return [
                    SimulationResult(initial.simulation_id, initial.iteration, "completed", {"value": 1}),
                    SimulationResult("dynamic-replacement", initial.iteration, "completed", {"value": 2}),
                ]

        def update(_state, observation, results):
            return StrategyUpdate(summary={"returned": len(results)})

        adaptive_strategy = CallbackStrategy(
            propose_experiment=lambda _state, iteration: ExperimentRequest("exp", iteration),
            propose_simulations=lambda _state, observation: [
                SimulationRequest("initial", observation.iteration, {})
            ],
            update=update,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = ClosedLoopRunner(
                campaign_id="adaptive",
                state_store=JsonStateStore(Path(directory) / "state.json"),
                acquisition=ProxyAcquisitionBackend(lambda request: {"x": 0}),
                simulation=AdaptiveSimulation(),
                strategy=adaptive_strategy,
            )
            state = runner.run(1)
            record = state["iterations"][0]
            self.assertEqual(len(record["simulation_requests"]), 1)
            self.assertEqual(len(record["simulation_results"]), 2)
            self.assertEqual(record["strategy_update"]["summary"]["returned"], 2)

    def test_adaptive_backend_must_still_return_every_initial_request(self):
        class DropsInitialSimulation:
            def submit(self, requests):
                return SimulationBatchHandle(
                    backend="adaptive-test",
                    handle_id="adaptive-0",
                    requests=tuple(requests),
                    metadata={"adaptive_requests": True},
                )

            def collect(self, handle):
                return [SimulationResult("replacement-only", 0, "completed", {"value": 2})]

        with tempfile.TemporaryDirectory() as directory:
            runner = ClosedLoopRunner(
                campaign_id="adaptive-missing",
                state_store=JsonStateStore(Path(directory) / "state.json"),
                acquisition=ProxyAcquisitionBackend(lambda request: {"x": 0}),
                simulation=DropsInitialSimulation(),
                strategy=strategy(),
            )
            with self.assertRaisesRegex(ValueError, "batch mismatch"):
                runner.run(1)


if __name__ == "__main__":
    unittest.main()
