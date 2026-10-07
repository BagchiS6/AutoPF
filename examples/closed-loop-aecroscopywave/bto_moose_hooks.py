"""Site-neutral command hook for BTO MOOSE chores."""

from __future__ import annotations

import json


def build_moose_command(request, work_directory, config):
    values = dict(request.parameters)
    mapping = dict(config["condition_override_map"])
    for key, value in request.condition.items():
        if key not in mapping:
            raise KeyError(f"No MOOSE override mapping configured for condition {key!r}.")
        values[mapping[key]] = value
    arguments = [f"{key}={value}" for key, value in sorted(values.items())]
    return [
        str(config["task_wrapper"]),
        str(work_directory),
        request.simulation_id,
        str(request.iteration),
        json.dumps(request.metadata, separators=(",", ":")),
        json.dumps(values, separators=(",", ":")),
        str(config["moose_app"]),
        "-i",
        str(config["base_input"]),
        *arguments,
    ]
