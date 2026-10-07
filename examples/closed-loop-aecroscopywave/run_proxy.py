#!/usr/bin/env python3
"""Run a complete restartable experiment--simulation loop without hardware."""

from __future__ import annotations

import argparse
from pathlib import Path

from autopf.closed_loop import ClosedLoopRunner, JsonStateStore
from autopf.closed_loop.backends import ProxyAcquisitionBackend, ProxySimulationBackend

from example_strategy import make_strategy, scaled_field


TRUE_HIDDEN_AMPLITUDE = 0.62


def acquire(request):
    voltage = float(request.payload["voltage_v"])
    duration = float(request.payload["duration_s"])
    return {
        "surface_uz": scaled_field(TRUE_HIDDEN_AMPLITUDE, voltage, duration),
        "voltage_v": voltage,
        "duration_s": duration,
        "metadata": {"condition": {"voltage_v": voltage, "duration_s": duration}},
    }


def simulate(request):
    amplitude = float(request.parameters["hidden_amplitude"])
    voltage = float(request.condition["voltage_v"])
    duration = float(request.condition["duration_s"])
    return {
        "surface_uz": scaled_field(amplitude, voltage, duration),
        "hidden_amplitude": amplitude,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--state", default="output/proxy_campaign_state.json")
    args = parser.parse_args()

    runner = ClosedLoopRunner(
        campaign_id="autopf-proxy-demo",
        state_store=JsonStateStore(Path(args.state)),
        acquisition=ProxyAcquisitionBackend(acquire),
        simulation=ProxySimulationBackend(simulate),
        strategy=make_strategy(),
        metadata={"acquisition_mode": "proxy", "true_hidden_amplitude": TRUE_HIDDEN_AMPLITUDE},
    )
    state = runner.run(args.iterations)
    print(f"status={state['status']} iterations={state['next_iteration']}")
    print(f"posterior={state['strategy_state']}")
    print(f"checkpoint={Path(args.state).resolve()}")


if __name__ == "__main__":
    main()
