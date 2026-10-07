"""JSON-safe records exchanged by the closed-loop engine and its adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


JsonMap = dict[str, Any]


def _mapping(value: Mapping[str, Any] | None) -> JsonMap:
    return dict(value or {})


@dataclass(frozen=True)
class ExperimentRequest:
    """One microscope action requested by the theory strategy."""

    request_id: str
    iteration: int
    job_type: str = "pfm_image"
    payload: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentRequest":
        return cls(
            request_id=str(value["request_id"]),
            iteration=int(value["iteration"]),
            job_type=str(value.get("job_type", "pfm_image")),
            payload=_mapping(value.get("payload")),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class AcquisitionHandle:
    """Durable identifier returned after an acquisition has been submitted."""

    backend: str
    handle_id: str
    request: ExperimentRequest
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcquisitionHandle":
        return cls(
            backend=str(value["backend"]),
            handle_id=str(value["handle_id"]),
            request=ExperimentRequest.from_dict(value["request"]),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class Observation:
    """Microscope result and portable references to any large arrays."""

    observation_id: str
    request_id: str
    iteration: int
    data: JsonMap = field(default_factory=dict)
    data_references: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        return cls(
            observation_id=str(value["observation_id"]),
            request_id=str(value["request_id"]),
            iteration=int(value["iteration"]),
            data=_mapping(value.get("data")),
            data_references=_mapping(value.get("data_references")),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class SimulationRequest:
    """One theory evaluation proposed after observing an experiment."""

    simulation_id: str
    iteration: int
    parameters: JsonMap
    condition: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationRequest":
        return cls(
            simulation_id=str(value["simulation_id"]),
            iteration=int(value["iteration"]),
            parameters=_mapping(value.get("parameters")),
            condition=_mapping(value.get("condition")),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class SimulationBatchHandle:
    """Durable handle for a submitted batch of theory evaluations."""

    backend: str
    handle_id: str
    requests: tuple[SimulationRequest, ...]
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        payload = asdict(self)
        payload["requests"] = [request.to_dict() for request in self.requests]
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationBatchHandle":
        return cls(
            backend=str(value["backend"]),
            handle_id=str(value["handle_id"]),
            requests=tuple(SimulationRequest.from_dict(row) for row in value.get("requests", [])),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class SimulationResult:
    """A completed simulation with inline summaries and external field paths."""

    simulation_id: str
    iteration: int
    status: str
    outputs: JsonMap = field(default_factory=dict)
    data_references: JsonMap = field(default_factory=dict)
    metadata: JsonMap = field(default_factory=dict)

    def to_dict(self) -> JsonMap:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationResult":
        return cls(
            simulation_id=str(value["simulation_id"]),
            iteration=int(value["iteration"]),
            status=str(value.get("status", "completed")),
            outputs=_mapping(value.get("outputs")),
            data_references=_mapping(value.get("data_references")),
            metadata=_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class StrategyUpdate:
    """Result of assimilating an observation and its theory evaluations."""

    summary: JsonMap = field(default_factory=dict)
    strategy_state: JsonMap = field(default_factory=dict)
    stop: bool = False

    def to_dict(self) -> JsonMap:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyUpdate":
        return cls(
            summary=_mapping(value.get("summary")),
            strategy_state=_mapping(value.get("strategy_state")),
            stop=bool(value.get("stop", False)),
        )
