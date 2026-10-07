"""Restartable experiment--simulation loops for AutoPF.

The core package is intentionally independent of microscope and scheduler
dependencies.  Instrument and simulation systems enter through small backend
interfaces, which makes the same campaign runnable against proxy data in CI and
against AEcroscopyWave plus MOOSE/MatEnsemble in production.
"""

from .engine import ClosedLoopRunner
from .interfaces import AcquisitionBackend, ClosedLoopStrategy, SimulationBackend
from .schema import (
    AcquisitionHandle,
    ExperimentRequest,
    Observation,
    SimulationBatchHandle,
    SimulationRequest,
    SimulationResult,
    StrategyUpdate,
)
from .state import JsonStateStore
from .objectives import ObjectiveEvaluator, ObjectiveMode, ObjectiveResult


def __getattr__(name):
    """Load the optional Torch production stack only when it is requested."""

    if name == "PODGaussianProcess":
        from .pod_gp import PODGaussianProcess

        return PODGaussianProcess
    if name in {"CandidateLibrary", "ProductionPODGPStrategy", "resolve_objective_mode"}:
        from .production import (
            CandidateLibrary,
            ProductionPODGPStrategy,
            resolve_objective_mode,
        )

        return {
            "CandidateLibrary": CandidateLibrary,
            "ProductionPODGPStrategy": ProductionPODGPStrategy,
            "resolve_objective_mode": resolve_objective_mode,
        }[name]
    raise AttributeError(name)

__all__ = [
    "AcquisitionBackend",
    "AcquisitionHandle",
    "ClosedLoopRunner",
    "ClosedLoopStrategy",
    "ExperimentRequest",
    "JsonStateStore",
    "Observation",
    "SimulationBackend",
    "SimulationBatchHandle",
    "SimulationRequest",
    "SimulationResult",
    "StrategyUpdate",
    "CandidateLibrary",
    "ObjectiveEvaluator",
    "ObjectiveMode",
    "ObjectiveResult",
    "PODGaussianProcess",
    "ProductionPODGPStrategy",
    "resolve_objective_mode",
]
