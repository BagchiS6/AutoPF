from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autopf.closed_loop import (
    ClosedLoopRunner,
    ExperimentRequest,
    JsonStateStore,
    SimulationRequest,
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


if __name__ == "__main__":
    unittest.main()
