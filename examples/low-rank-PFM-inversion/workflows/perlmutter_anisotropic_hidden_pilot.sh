#!/bin/bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -J bto_anisveg
#SBATCH -o logs/bto_anisveg_%j.out
#SBATCH -e logs/bto_anisveg_%j.err

set -euo pipefail

export MPICH_GPU_SUPPORT_ENABLED=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-simbo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export SIM_BO_SKIP_COMPLETED_RUNS="${SIM_BO_SKIP_COMPLETED_RUNS:-1}"

AUTOPF_ENV="${AUTOPF_ENV:-/global/cfs/cdirs/m5014/PhaseField/autopf_env}"
PYTHON="$AUTOPF_ENV/bin/python"
MATENSEMBLE="$AUTOPF_ENV/bin/matensemble-launcher"
export PATH="$AUTOPF_ENV/bin:$PATH"

if [[ -n "${CAMPAIGN_DIR:-}" ]]; then
  script_dir="$CAMPAIGN_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_manifest_round.py" ]]; then
  script_dir="$SLURM_SUBMIT_DIR"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$script_dir"
mkdir -p logs condition_gp_active

ROUND="${ROUND:-anisotropic_hidden_pilot_000}"
MANIFEST="$script_dir/round_${ROUND}/manifest.json"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file not found: $path" >&2
    exit 2
  fi
}

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "ERROR: required executable not found or not executable: $path" >&2
    exit 2
  fi
}

require_executable "$PYTHON"
require_executable "$MATENSEMBLE"
require_file "BTO_DW_anisotropic_hidden_physics_zdecay_ic.i"
require_file "build_anisotropic_hidden_physics_manifest.py"
require_file "run_manifest_round.py"
require_file "summarize_anisotropic_hidden_pilot.py"

echo "========================================"
echo "BTO ANISOTROPIC HIDDEN-PHYSICS PILOT"
echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node list:    ${SLURM_NODELIST:-local}"
echo "Campaign dir: $script_dir"
echo "Round:        $ROUND"
echo "Manifest:     $MANIFEST"
echo "Python:       $PYTHON"
"$PYTHON" --version
echo "MatEnsemble:  $MATENSEMBLE"
echo "Skip done:    $SIM_BO_SKIP_COMPLETED_RUNS"
echo "========================================"

if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON" build_anisotropic_hidden_physics_manifest.py \
    --campaign-dir "$script_dir" \
    --round "$ROUND" \
    --base-input BTO_DW_anisotropic_hidden_physics_zdecay_ic.i
fi

"$MATENSEMBLE" run_manifest_round.py --manifest "$MANIFEST"

"$PYTHON" summarize_anisotropic_hidden_pilot.py \
  --campaign-dir "$script_dir" \
  --round "$ROUND"

"$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
obs_path = manifest.parent / "observations.json"
if not obs_path.exists():
    raise SystemExit(f"observations not found: {obs_path}")
rows = json.loads(obs_path.read_text(encoding="utf-8"))
best = sorted(rows, key=lambda row: row["objective"])[:10]
print("Top anisotropic hidden-physics candidates:")
for row in best:
    params = row["parameters"]
    print(
        f"  {row['candidate_id']}: nMSE={row['objective']:.6e}, "
        f"conditions={row.get('n_conditions', 'NA')}, "
        f"screen={params.get('spatial_screen_amp', 0):.4g}, "
        f"eps_xx={params.get('anis_vegard_xx_amp', 0):.4g}, "
        f"eps_yy={params.get('anis_vegard_yy_amp', 0):.4g}, "
        f"eps_zz={params.get('anis_vegard_zz_amp', 0):.4g}, "
        f"eps_xy={params.get('anis_vegard_xy_amp', 0):.4g}, "
        f"eps_xz={params.get('anis_vegard_xz_amp', 0):.4g}, "
        f"eps_yz={params.get('anis_vegard_yz_amp', 0):.4g}, "
        f"flex={params.get('flexo_proxy_amp', 0):.4g}, "
        f"flex_grad={params.get('flexo_proxy_grad_amp', 0):.4g}"
    )
PY
