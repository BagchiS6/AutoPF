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
]
