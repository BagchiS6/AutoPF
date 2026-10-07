"""AEcroscopyWave REST adapter for AutoPF closed-loop campaigns.

The adapter targets the public server contract in AEcroscopyWave 0.1.21:

* ``POST /api/v1/jobs/enqueue``
* ``GET  /api/v1/jobs/{job_id}/result``

Large arrays remain in AEcroscopyWave's Tiled catalog.  AutoPF records their
portable references and lets the scientific strategy resolve them using the
site's preferred Tiled client.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

from ..schema import AcquisitionHandle, ExperimentRequest, Observation


class AEcroscopyWaveError(RuntimeError):
    """Raised when an AEcroscopyWave job cannot be submitted or collected."""


ResultResolver = Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


def _split_result_references(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_keys = {
        "tiled_uri",
        "tiled_run_key",
        "tiled_array_keys",
        "h5_path",
        "ibw_path",
        "sidpy_h5_path",
        "output_path",
    }
    references = {key: value for key, value in result.items() if key in reference_keys}
    data = {key: value for key, value in result.items() if key not in reference_keys}
    return data, references


class TiledArrayResolver:
    """Load one AEcroscopyWave-published array from the returned Tiled URI.

    ``transform`` is the experiment-specific calibration boundary. For PFM it
    can combine amplitude, phase, or other channels into the field expected by
    the theory model. Without a transform, the selected array is only aliased
    to ``output_key``; AutoPF does not claim that raw amplitude is displacement.
    """

    def __init__(
        self,
        *,
        array_key: str,
        output_key: str = "surface_uz",
        api_key: str | None = None,
        transform: Callable[[Any, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.array_key = array_key
        self.output_key = output_key
        self.api_key = api_key
        self.transform = transform

    def __call__(self, result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        data, references = _split_result_references(result)
        uri = references.get("tiled_uri")
        if not uri:
            raise AEcroscopyWaveError("AEcroscopyWave result does not contain a tiled_uri.")
        try:
            from tiled.client import from_uri
        except ImportError as exc:
            raise AEcroscopyWaveError(
                "Loading AEcroscopyWave arrays requires AutoPF's 'tiled' extra."
            ) from exc
        node = from_uri(str(uri), api_key=self.api_key or None)
        try:
            array = node[self.array_key].read()
        except Exception as exc:
            available = references.get("tiled_array_keys", [])
            raise AEcroscopyWaveError(
                f"Tiled array {self.array_key!r} could not be loaded; advertised keys={available}."
            ) from exc
        calibrated = self.transform(array, result) if self.transform else array
        data[self.output_key] = calibrated.tolist() if hasattr(calibrated, "tolist") else calibrated
        return data, references


class AEcroscopyWaveAcquisitionBackend:
    """Submit microscope jobs to an AEcroscopyWave cloud server.

    Parameters
    ----------
    server_url:
        Base URL of the ``afw-server`` service.
    instrument_id:
        Registered AEcroscopyWave instrument identifier.
    secret:
        Server bearer token.  It is kept in memory and never serialized into an
        AutoPF campaign checkpoint.
    result_resolver:
        Optional hook that can replace Tiled references with loaded arrays.
        The default keeps scalar metadata inline and records Tiled identifiers.
    """

    backend_name = "aecroscopywave-rest-v0.1"

    def __init__(
        self,
        *,
        server_url: str,
        instrument_id: str,
        secret: str,
        poll_interval_seconds: float = 5.0,
        timeout_seconds: float = 3600.0,
        request_timeout_seconds: float = 30.0,
        result_resolver: ResultResolver | None = None,
    ) -> None:
        if not secret:
            raise ValueError("AEcroscopyWave secret must not be empty.")
        self.server_url = server_url.rstrip("/")
        self.instrument_id = instrument_id
        self._secret = secret
        self.poll_interval_seconds = max(float(poll_interval_seconds), 0.01)
        self.timeout_seconds = max(float(timeout_seconds), 0.01)
        self.request_timeout_seconds = max(float(request_timeout_seconds), 0.01)
        self.result_resolver = result_resolver or _split_result_references

    def _request_json(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        encoded = None if body is None else json.dumps(dict(body)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=encoded,
            method=method,
            headers={
                "Authorization": f"Bearer {self._secret}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                payload = response.read().decode("utf-8")
                return int(response.status), json.loads(payload or "{}")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw or "{}")
            except json.JSONDecodeError:
                payload = {"detail": raw}
            return int(exc.code), payload
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AEcroscopyWaveError(f"{method} {path} failed: {exc}") from exc

    def submit(self, request: ExperimentRequest) -> AcquisitionHandle:
        status, payload = self._request_json(
            "POST",
            "/api/v1/jobs/enqueue",
            {
                "instrument_id": self.instrument_id,
                "job_type": request.job_type,
                "payload": request.payload,
            },
        )
        if status != 200 or not payload.get("job_id"):
            raise AEcroscopyWaveError(
                f"AEcroscopyWave enqueue failed with HTTP {status}: {payload}"
            )
        return AcquisitionHandle(
            backend=self.backend_name,
            handle_id=str(payload["job_id"]),
            request=request,
            metadata={
                "server_url": self.server_url,
                "instrument_id": self.instrument_id,
                "remote_status": str(payload.get("status", "queued")),
            },
        )

    def collect(self, handle: AcquisitionHandle) -> Observation:
        if handle.backend != self.backend_name:
            raise ValueError(f"Handle backend {handle.backend!r} does not match {self.backend_name!r}.")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            status, payload = self._request_json(
                "GET", f"/api/v1/jobs/{handle.handle_id}/result"
            )
            if status == 200:
                result = dict(payload)
                data, references = self.result_resolver(result)
                return Observation(
                    observation_id=handle.handle_id,
                    request_id=handle.request.request_id,
                    iteration=handle.request.iteration,
                    data=data,
                    data_references=references,
                    metadata={
                        "backend": self.backend_name,
                        "instrument_id": self.instrument_id,
                        "job_type": handle.request.job_type,
                        **handle.request.metadata,
                    },
                )
            if status != 409:
                raise AEcroscopyWaveError(
                    f"AEcroscopyWave result retrieval failed with HTTP {status}: {payload}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting {self.timeout_seconds:g} s for AEcroscopyWave job {handle.handle_id}."
                )
            time.sleep(self.poll_interval_seconds)
