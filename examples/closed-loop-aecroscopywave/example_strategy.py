"""Small UQ strategy used by both proxy and live-acquisition examples.

This is deliberately lightweight: it demonstrates the adapter and restart
contracts, not the production POD--GP model.  Replace these callbacks with the
campaign's POD--GP/Thompson-sampling functions without changing the runner.
"""

from __future__ import annotations

import math

from autopf.closed_loop import ExperimentRequest, SimulationRequest, StrategyUpdate
from autopf.closed_loop.strategies import CallbackStrategy


def basis_field(size: int = 8) -> list[list[float]]:
    center = (size - 1) / 2
    return [
        [math.exp(-((x - center) ** 2 + (y - center) ** 2) / (0.18 * size * size)) for x in range(size)]
        for y in range(size)
    ]


def scaled_field(amplitude: float, voltage_v: float, pulse_s: float) -> list[list[float]]:
    gain = abs(voltage_v) * math.sqrt(max(pulse_s, 1.0e-6))
    return [[amplitude * gain * value for value in row] for row in basis_field()]


def mean_squared_error(left: list[list[float]], right: list[list[float]]) -> float:
    pairs = [(a, b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b)]
    return sum((a - b) ** 2 for a, b in pairs) / max(len(pairs), 1)


def make_strategy(scan_size_nm: float = 350.0, num_lines: int = 64) -> CallbackStrategy:
    conditions = [
        {"voltage_v": 3.0, "duration_s": 0.10},
        {"voltage_v": 5.0, "duration_s": 0.30},
        {"voltage_v": 7.8, "duration_s": 0.55},
        {"voltage_v": -5.0, "duration_s": 0.30},
    ]

    def propose_experiment(state, iteration):
        condition = conditions[iteration % len(conditions)]
        return ExperimentRequest(
            request_id=f"pfm-{iteration:04d}",
            iteration=iteration,
            job_type="pfm_image",
            payload={
                "scan_size_nm": scan_size_nm,
                "scan_rate_hz": 1.0,
                "num_lines": num_lines,
                "base_filename": f"autopf_{state['campaign_id']}_{iteration:04d}_",
                # The pulse is retained as scientific condition metadata.  A
                # site workflow can enqueue dc_pulse before pfm_image if needed.
                **condition,
            },
            metadata={"condition": condition},
        )

    def propose_simulations(state, observation):
        posterior = state.get("strategy_state", {})
        center = float(posterior.get("posterior_mean", 0.5))
        scale = float(posterior.get("posterior_std", 0.35))
        values = sorted({max(0.0, min(1.0, center + offset * scale)) for offset in (-1.5, -0.75, 0, 0.75, 1.5)})
        condition = dict(observation.metadata.get("condition", {}))
        if not condition:
            condition = {
                "voltage_v": observation.data.get("voltage_v", 0.0),
                "duration_s": observation.data.get("duration_s", 0.1),
            }
        return [
            SimulationRequest(
                simulation_id=f"sim-{observation.iteration:04d}-{index:02d}",
                iteration=observation.iteration,
                parameters={"hidden_amplitude": value},
                condition=condition,
            )
            for index, value in enumerate(values)
        ]

    def update(state, observation, simulations):
        experimental = observation.data.get("surface_uz")
        if experimental is None:
            raise ValueError(
                "The example strategy needs observation.data['surface_uz']. "
                "For live AEcroscopyWave, provide a result_resolver that loads the Tiled PFM array."
            )
        scored = []
        for result in simulations:
            field = result.outputs["surface_uz"]
            amplitude = float(result.outputs["hidden_amplitude"])
            scored.append((mean_squared_error(experimental, field), amplitude))
        scale = max(sorted(loss for loss, _ in scored)[len(scored) // 2], 1.0e-10)
        raw = [(math.exp(-loss / scale), amplitude, loss) for loss, amplitude in scored]
        norm = sum(weight for weight, _, _ in raw)
        weights = [(weight / norm, amplitude, loss) for weight, amplitude, loss in raw]
        mean = sum(weight * amplitude for weight, amplitude, _ in weights)
        variance = sum(weight * (amplitude - mean) ** 2 for weight, amplitude, _ in weights)
        std = math.sqrt(max(variance, 0.0))
        best = min(scored)
        return StrategyUpdate(
            summary={"best_loss": best[0], "map_hidden_amplitude": best[1]},
            strategy_state={"posterior_mean": mean, "posterior_std": std},
            stop=std < 0.02,
        )

    return CallbackStrategy(
        propose_experiment=propose_experiment,
        propose_simulations=propose_simulations,
        update=update,
    )
