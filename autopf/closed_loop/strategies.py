"""Small strategy adapters for embedding existing UQ and surrogate code."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .schema import ExperimentRequest, Observation, SimulationRequest, SimulationResult, StrategyUpdate


class CallbackStrategy:
    """Connect existing POD--GP/TS functions to the closed-loop engine.

    This adapter deliberately imposes no model implementation.  Production
    campaigns can retain their trained surrogate and posterior objects while
    the runner supplies durable experiment/simulation orchestration.
    """

    def __init__(
        self,
        *,
        propose_experiment: Callable[[dict[str, Any], int], ExperimentRequest],
        propose_simulations: Callable[[dict[str, Any], Observation], Sequence[SimulationRequest]],
        update: Callable[
            [dict[str, Any], Observation, Sequence[SimulationResult]], StrategyUpdate
        ],
    ) -> None:
        self._propose_experiment = propose_experiment
        self._propose_simulations = propose_simulations
        self._update = update

    def propose_experiment(self, state: dict[str, Any], iteration: int) -> ExperimentRequest:
        return self._propose_experiment(state, iteration)

    def propose_simulations(
        self, state: dict[str, Any], observation: Observation
    ) -> Sequence[SimulationRequest]:
        return self._propose_simulations(state, observation)

    def update(
        self,
        state: dict[str, Any],
        observation: Observation,
        simulations: Sequence[SimulationResult],
    ) -> StrategyUpdate:
        return self._update(state, observation, simulations)
