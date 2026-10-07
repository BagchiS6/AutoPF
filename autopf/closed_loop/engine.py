"""Restartable state machine for repeated experiment--simulation iterations."""

from __future__ import annotations

from typing import Any

from .interfaces import AcquisitionBackend, ClosedLoopStrategy, SimulationBackend
from .schema import (
    AcquisitionHandle,
    ExperimentRequest,
    Observation,
    SimulationBatchHandle,
    SimulationRequest,
    SimulationResult,
)
from .state import JsonStateStore, utc_now


_PHASES = (
    "created",
    "experiment_planned",
    "acquisition_submitted",
    "acquired",
    "simulations_planned",
    "simulations_submitted",
    "simulated",
    "updated",
    "complete",
)


class ClosedLoopRunner:
    """Execute a campaign and checkpoint after every external action.

    Submission and collection are separate transitions.  Consequently, if the
    process stops while waiting for an instrument or HPC batch, the durable
    remote handle is reused on restart instead of submitting a duplicate job.
    """

    def __init__(
        self,
        *,
        campaign_id: str,
        state_store: JsonStateStore,
        acquisition: AcquisitionBackend,
        simulation: SimulationBackend,
        strategy: ClosedLoopStrategy,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.campaign_id = campaign_id
        self.state_store = state_store
        self.acquisition = acquisition
        self.simulation = simulation
        self.strategy = strategy
        self.metadata = dict(metadata or {})

    def run(self, max_iterations: int) -> dict[str, Any]:
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        state = self.state_store.initialize(self.campaign_id, self.metadata)
        if state.get("status") == "complete":
            if (
                state.get("stop_reason") == "max_iterations"
                and int(state.get("next_iteration", 0)) < max_iterations
            ):
                state["status"] = "running"
                state.pop("stop_reason", None)
                self.state_store.save(state)
            else:
                return state

        while int(state["next_iteration"]) < max_iterations:
            iteration = int(state["next_iteration"])
            record = self._iteration_record(state, iteration)
            self._advance(state, record)
            if record.get("stop", False):
                state["status"] = "complete"
                state["stop_reason"] = "strategy"
                self.state_store.save(state)
                return state

        state["status"] = "complete"
        state["stop_reason"] = "max_iterations"
        self.state_store.save(state)
        return state

    def _iteration_record(self, state: dict[str, Any], iteration: int) -> dict[str, Any]:
        for record in state["iterations"]:
            if int(record["iteration"]) == iteration:
                return record
        record = {
            "iteration": iteration,
            "phase": "created",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        state["iterations"].append(record)
        self.state_store.save(state)
        return record

    def _checkpoint(self, state: dict[str, Any], record: dict[str, Any], phase: str) -> None:
        if phase not in _PHASES:
            raise ValueError(f"Unknown phase: {phase}")
        record["phase"] = phase
        record["updated_at"] = utc_now()
        self.state_store.save(state)

    def _advance(self, state: dict[str, Any], record: dict[str, Any]) -> None:
        phase = record["phase"]
        iteration = int(record["iteration"])

        if phase == "created":
            request = self.strategy.propose_experiment(state, iteration)
            if request.iteration != iteration:
                raise ValueError("Strategy returned an experiment request for the wrong iteration.")
            record["experiment_request"] = request.to_dict()
            self._checkpoint(state, record, "experiment_planned")
            phase = record["phase"]

        if phase == "experiment_planned":
            request = ExperimentRequest.from_dict(record["experiment_request"])
            handle = self.acquisition.submit(request)
            record["acquisition_handle"] = handle.to_dict()
            self._checkpoint(state, record, "acquisition_submitted")
            phase = record["phase"]

        if phase == "acquisition_submitted":
            handle = AcquisitionHandle.from_dict(record["acquisition_handle"])
            observation = self.acquisition.collect(handle)
            record["observation"] = observation.to_dict()
            self._checkpoint(state, record, "acquired")
            phase = record["phase"]

        if phase == "acquired":
            observation = Observation.from_dict(record["observation"])
            requests = tuple(self.strategy.propose_simulations(state, observation))
            if any(request.iteration != iteration for request in requests):
                raise ValueError("Strategy returned a simulation request for the wrong iteration.")
            record["simulation_requests"] = [request.to_dict() for request in requests]
            self._checkpoint(state, record, "simulations_planned")
            phase = record["phase"]

        if phase == "simulations_planned":
            requests = [SimulationRequest.from_dict(row) for row in record["simulation_requests"]]
            handle = self.simulation.submit(requests)
            record["simulation_handle"] = handle.to_dict()
            self._checkpoint(state, record, "simulations_submitted")
            phase = record["phase"]

        if phase == "simulations_submitted":
            handle = SimulationBatchHandle.from_dict(record["simulation_handle"])
            results = tuple(self.simulation.collect(handle))
            requested = {row["simulation_id"] for row in record["simulation_requests"]}
            returned = {result.simulation_id for result in results}
            if requested != returned:
                raise ValueError(f"Simulation batch mismatch: requested={requested}, returned={returned}")
            record["simulation_results"] = [result.to_dict() for result in results]
            self._checkpoint(state, record, "simulated")
            phase = record["phase"]

        if phase == "simulated":
            observation = Observation.from_dict(record["observation"])
            results = [SimulationResult.from_dict(row) for row in record["simulation_results"]]
            update = self.strategy.update(state, observation, results)
            record["strategy_update"] = update.to_dict()
            state["strategy_state"] = dict(update.strategy_state)
            record["stop"] = bool(update.stop)
            self._checkpoint(state, record, "updated")
            phase = record["phase"]

        if phase == "updated":
            state["next_iteration"] = iteration + 1
            self._checkpoint(state, record, "complete")
