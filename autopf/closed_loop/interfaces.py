"""Backend and strategy protocols for closed-loop AutoPF campaigns."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .schema import (
    AcquisitionHandle,
    ExperimentRequest,
    Observation,
    SimulationBatchHandle,
    SimulationRequest,
    SimulationResult,
    StrategyUpdate,
)


class AcquisitionBackend(Protocol):
    """Submit and collect an experiment without assuming a specific microscope."""

    def submit(self, request: ExperimentRequest) -> AcquisitionHandle:
        ...

    def collect(self, handle: AcquisitionHandle) -> Observation:
        ...


class SimulationBackend(Protocol):
    """Submit and collect a batch of theory evaluations."""

    def submit(self, requests: Sequence[SimulationRequest]) -> SimulationBatchHandle:
        ...

    def collect(self, handle: SimulationBatchHandle) -> Sequence[SimulationResult]:
        ...


class ClosedLoopStrategy(Protocol):
    """Scientific policy connecting observations, simulations, and UQ state."""

    def propose_experiment(self, campaign_state: dict[str, Any], iteration: int) -> ExperimentRequest:
        ...

    def propose_simulations(
        self,
        campaign_state: dict[str, Any],
        observation: Observation,
    ) -> Sequence[SimulationRequest]:
        ...

    def update(
        self,
        campaign_state: dict[str, Any],
        observation: Observation,
        simulations: Sequence[SimulationResult],
    ) -> StrategyUpdate:
        ...
