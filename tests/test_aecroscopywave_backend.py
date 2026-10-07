from __future__ import annotations

import io
import json
import sys
import types
import unittest
import urllib.error
from unittest.mock import patch

from autopf.closed_loop import ExperimentRequest
from autopf.closed_loop.backends import AEcroscopyWaveAcquisitionBackend, TiledArrayResolver


class Response:
    def __init__(self, status, payload):
        self.status = status
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class AEcroscopyWaveBackendTests(unittest.TestCase):
    def test_enqueue_poll_and_tiled_references(self):
        calls = []
        responses = [
            Response(200, {"job_id": "job-1", "status": "queued"}),
            urllib.error.HTTPError(
                "http://server/api/v1/jobs/job-1/result",
                409,
                "pending",
                {},
                io.BytesIO(b'{"detail":"not complete"}'),
            ),
            Response(
                200,
                {
                    "pfm_amplitude": "<array shape=(64, 64) dtype=float64>",
                    "tiled_uri": "http://tiled/runs/job-1",
                    "tiled_run_key": "job-1",
                    "tiled_array_keys": ["pfm_amplitude", "pfm_phase"],
                },
            ),
        ]

        def urlopen(request, timeout):
            calls.append((request, timeout))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        backend = AEcroscopyWaveAcquisitionBackend(
            server_url="http://server",
            instrument_id="cypher-1",
            secret="secret",
            poll_interval_seconds=0.001,
            timeout_seconds=1,
        )
        request = ExperimentRequest("request-1", 0, payload={"scan_size_nm": 350, "scan_rate_hz": 1, "num_lines": 64})
        with patch("urllib.request.urlopen", side_effect=urlopen):
            handle = backend.submit(request)
            observation = backend.collect(handle)

        self.assertEqual(handle.handle_id, "job-1")
        self.assertNotIn("secret", json.dumps(handle.to_dict()).lower())
        self.assertEqual(observation.data_references["tiled_run_key"], "job-1")
        self.assertIn("pfm_amplitude", observation.data)
        self.assertEqual(calls[0][0].get_header("Authorization"), "Bearer secret")

    def test_tiled_resolver_loads_selected_array(self):
        class Array:
            def read(self):
                return self

            def tolist(self):
                return [[1.0, 2.0], [3.0, 4.0]]

        client = types.ModuleType("tiled.client")
        client.from_uri = lambda uri, api_key=None: {"pfm_amplitude": Array()}
        tiled = types.ModuleType("tiled")
        tiled.client = client
        resolver = TiledArrayResolver(array_key="pfm_amplitude", output_key="surface_uz")
        with patch.dict(sys.modules, {"tiled": tiled, "tiled.client": client}):
            data, references = resolver(
                {
                    "tiled_uri": "http://tiled/runs/job-1",
                    "tiled_array_keys": ["pfm_amplitude"],
                }
            )
        self.assertEqual(data["surface_uz"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(references["tiled_uri"], "http://tiled/runs/job-1")


if __name__ == "__main__":
    unittest.main()
