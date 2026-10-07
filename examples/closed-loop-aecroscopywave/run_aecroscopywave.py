#!/usr/bin/env python3
"""Run the same loop against an AEcroscopyWave instrument and proxy theory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from autopf.closed_loop import ClosedLoopRunner, JsonStateStore
from autopf.closed_loop.backends import (
    AEcroscopyWaveAcquisitionBackend,
    ProxySimulationBackend,
    TiledArrayResolver,
)

from example_strategy import make_strategy, scaled_field


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
    parser.add_argument("--state", default="output/live_campaign_state.json")
    parser.add_argument("--server-url", default=os.environ.get("AFW_CLOUD_SERVER_URL", ""))
    parser.add_argument("--instrument-id", default=os.environ.get("AFW_INSTRUMENT_ID", ""))
    parser.add_argument("--tiled-array-key", default="pfm_amplitude")
    args = parser.parse_args()
    secret = os.environ.get("AFW_AGENT_SECRET", "")
    if not args.server_url or not args.instrument_id or not secret:
        raise SystemExit("Set AFW_CLOUD_SERVER_URL, AFW_INSTRUMENT_ID, and AFW_AGENT_SECRET.")

    acquisition = AEcroscopyWaveAcquisitionBackend(
        server_url=args.server_url,
        instrument_id=args.instrument_id,
        secret=secret,
        result_resolver=TiledArrayResolver(
            array_key=args.tiled_array_key,
            output_key="surface_uz",
            api_key=os.environ.get("AFW_TILED_API_KEY"),
            # Add transform=... for the experiment's calibrated PFM-to-u_z map.
        ),
    )
    runner = ClosedLoopRunner(
        campaign_id="autopf-live-demo",
        state_store=JsonStateStore(Path(args.state)),
        acquisition=acquisition,
        simulation=ProxySimulationBackend(simulate),
        strategy=make_strategy(),
        metadata={"acquisition_mode": "aecroscopywave"},
    )
    state = runner.run(args.iterations)
    print(f"status={state['status']} iterations={state['next_iteration']}")
    print(f"posterior={state['strategy_state']}")


if __name__ == "__main__":
    main()
