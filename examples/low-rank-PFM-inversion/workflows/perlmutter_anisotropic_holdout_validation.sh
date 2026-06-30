#!/bin/bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -J bto_anis_hold
#SBATCH -o logs/bto_anis_hold_%j.out
#SBATCH -e logs/bto_anis_hold_%j.err

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

ROUND="${ROUND:-anisotropic_holdout_full_000}"
POLICY="${POLICY:-full30}"
PILOT_ROUND="${PILOT_ROUND:-anisotropic_hidden_pilot_000}"
MANIFEST="$script_dir/round_${ROUND}/manifest.json"
ANIS_CANDIDATE_ID="r${ROUND}_cand02"

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
require_file "build_anisotropic_holdout_validation_manifest.py"
require_file "run_manifest_round.py"
require_file "summarize_anisotropic_holdout_validation.py"
require_file "paper_figures/make_holdout_prediction_residual_panels.py"

echo "========================================"
echo "BTO ANISOTROPIC HIDDEN-PHYSICS HOLDOUT"
echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node list:    ${SLURM_NODELIST:-local}"
echo "Campaign dir: $script_dir"
echo "Round:        $ROUND"
echo "Policy:       $POLICY"
echo "Pilot round:  $PILOT_ROUND"
echo "Manifest:     $MANIFEST"
echo "Python:       $PYTHON"
"$PYTHON" --version
echo "MatEnsemble:  $MATENSEMBLE"
echo "Skip done:    $SIM_BO_SKIP_COMPLETED_RUNS"
echo "========================================"

if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON" build_anisotropic_holdout_validation_manifest.py \
    --campaign-dir "$script_dir" \
    --round "$ROUND" \
    --policy "$POLICY" \
    --pilot-round "$PILOT_ROUND" \
    --base-input BTO_DW_anisotropic_hidden_physics_zdecay_ic.i
fi

"$MATENSEMBLE" run_manifest_round.py --manifest "$MANIFEST"

"$PYTHON" summarize_anisotropic_holdout_validation.py \
  --campaign-dir "$script_dir" \
  --round "$ROUND"

"$PYTHON" paper_figures/make_holdout_prediction_residual_panels.py \
  --campaign-dir "$script_dir" \
  --round "$ROUND" \
  --config sim_bo_config_corrected_ic_posterior_guided.json \
  --candidate-id "$ANIS_CANDIDATE_ID" \
  --stem-suffix "_anisotropic_shear_yz_holdout" \
  --title-suffix "(anisotropic shear hidden physics)"

"$PYTHON" paper_figures/make_holdout_prediction_residual_panels.py \
  --campaign-dir "$script_dir" \
  --round "$ROUND" \
  --config sim_bo_config_corrected_ic_posterior_guided.json \
  --stem-suffix "_anisotropic_best_holdout" \
  --title-suffix "(best holdout candidate)"

echo "========================================"
echo "Anisotropic holdout validation complete."
echo "Summary: $script_dir/anisotropic_hidden_analysis/$ROUND/summary.json"
echo "Residual panels: $script_dir/paper_figures/generated/*anisotropic*holdout*.pdf"
echo "========================================"
