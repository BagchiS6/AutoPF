#!/usr/bin/env python3
"""Run an existing BO manifest through AutoPF/MatEnsemble and score it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_bo_loop import compute_round_losses, run_round


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument(
        "--observations",
        default="",
        help="Output observations JSON. Defaults to <manifest-dir>/observations.json.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only compute observations from existing outputs; do not launch MOOSE.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.score_only:
        run_round(manifest)
    observations = compute_round_losses(manifest)

    out_path = Path(args.observations).resolve() if args.observations else manifest_path.parent / "observations.json"
    out_path.write_text(json.dumps(observations, indent=2), encoding="utf-8")
    print(f"Wrote observations: {out_path}")
    for obs in sorted(observations, key=lambda row: row["objective"]):
        params = obs["parameters"]
        print(
            f"{obs['candidate_id']}: "
            f"g11={params['g11']:.6g} g12={params['g12']:.6g} g44={params['g44']:.6g} "
            f"nMSE={obs['objective']:.6e} conditions={obs['n_conditions']}"
        )


if __name__ == "__main__":
    main()
