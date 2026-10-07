"""Deterministic proxy backends with the same contracts as live systems."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, Union

from ..schema import (
    AcquisitionHandle,
    ExperimentRequest,
    Observation,
    SimulationBatchHandle,
    SimulationRequest,
    SimulationResult,
)


AcquisitionFunction = Callable[[ExperimentRequest], Union[Mapping[str, Any], Observation]]
SimulationFunction = Callable[[SimulationRequest], Union[Mapping[str, Any], SimulationResult]]


class ProxyAcquisitionBackend:
    """In-process acquisition backend for digital twins and replay datasets."""

    backend_name = "proxy-acquisition-v1"

    def __init__(self, acquire: AcquisitionFunction) -> None:
        self._acquire = acquire
        self._requests: dict[str, ExperimentRequest] = {}

    def submit(self, request: ExperimentRequest) -> AcquisitionHandle:
        self._requests[request.request_id] = request
        return AcquisitionHandle(
            backend=self.backend_name,
            handle_id=request.request_id,
            request=request,
            metadata={"mode": "proxy"},
        )

    def collect(self, handle: AcquisitionHandle) -> Observation:
        request = self._requests.get(handle.handle_id, handle.request)
        result = self._acquire(request)
        if isinstance(result, Observation):
            return result
        payload = dict(result)
        references = dict(payload.pop("data_references", {}))
        metadata = dict(payload.pop("metadata", {}))
        return Observation(
            observation_id=f"proxy-observation-{request.iteration:04d}",
            request_id=request.request_id,
            iteration=request.iteration,
            data=payload,
            data_references=references,
            metadata={"backend": self.backend_name, **request.metadata, **metadata},
        )


class ProxySimulationBackend:
    """In-process simulation backend for examples and workflow validation."""

    backend_name = "proxy-simulation-v1"

    def __init__(self, simulate: SimulationFunction) -> None:
        self._simulate = simulate
        self._batches: dict[str, tuple[SimulationRequest, ...]] = {}
        self.submit_count = 0

    def submit(self, requests: Sequence[SimulationRequest]) -> SimulationBatchHandle:
        rows = tuple(requests)
        iteration = rows[0].iteration if rows else 0
        handle_id = f"proxy-simulation-batch-{iteration:04d}"
        self._batches[handle_id] = rows
        self.submit_count += 1
        return SimulationBatchHandle(
            backend=self.backend_name,
            handle_id=handle_id,
            requests=rows,
            metadata={"mode": "proxy"},
        )

    def collect(self, handle: SimulationBatchHandle) -> Sequence[SimulationResult]:
        requests = self._batches.get(handle.handle_id, handle.requests)
        results = []
        for request in requests:
            result = self._simulate(request)
            if isinstance(result, SimulationResult):
                results.append(result)
                continue
            payload = dict(result)
            references = dict(payload.pop("data_references", {}))
            metadata = dict(payload.pop("metadata", {}))
            status = str(payload.pop("status", "completed"))
            results.append(
                SimulationResult(
                    simulation_id=request.simulation_id,
                    iteration=request.iteration,
                    status=status,
                    outputs=payload,
                    data_references=references,
                    metadata={"backend": self.backend_name, **metadata},
                )
            )
        return results
