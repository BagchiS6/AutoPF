#!/bin/bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 16:00:00
#SBATCH -N 16
#SBATCH -J bto_icpost
#SBATCH -o logs/bto_icpost_%j.out
#SBATCH -e logs/bto_icpost_%j.err

set -euo pipefail

export MPICH_GPU_SUPPORT_ENABLED=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-simbo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

AUTOPF_ENV="${AUTOPF_ENV:-/global/cfs/cdirs/m5014/PhaseField/autopf_env}"
export PATH="$AUTOPF_ENV/bin:$PATH"
PYTHON="$AUTOPF_ENV/bin/python"
MATENSEMBLE="$AUTOPF_ENV/bin/matensemble-launcher"

if [[ -n "${CAMPAIGN_DIR:-}" ]]; then
  _script_dir="$CAMPAIGN_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/build_posterior_guided_corrected_ic_config.py" ]]; then
  _script_dir="$SLURM_SUBMIT_DIR"
else
  _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$_script_dir"
mkdir -p logs condition_gp_active corrected_ic_posterior_guided_reanalysis

PREFIX="${PREFIX:-corrected_ic_postguided}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-sim_bo_config_corrected_ic_posterior_guided.json}"
BASE_INPUT="${BASE_INPUT:-BTO_DW_spatial_hidden_physics_zdecay_ic.i}"
ANCHOR_LIMIT="${ANCHOR_LIMIT:-24}"
REFINE_ROUNDS="${REFINE_ROUNDS:-3}"
HOLDOUT_POLICY="${HOLDOUT_POLICY:-full30}"
POST_ANALYSIS_DIR="${POST_ANALYSIS_DIR:-corrected_ic_posterior_guided_reanalysis}"

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
require_file "$BASE_INPUT"
require_file "run_manifest_round.py"
require_file "select_best_observation_params.py"
require_file "build_posterior_guided_corrected_ic_config.py"
require_file "build_corrected_ic_anchor_manifest.py"
require_file "build_hidden_physics_anchor_manifest.py"
require_file "build_spatial_hidden_physics_pilot_manifest.py"
require_file "build_spatial_refinement_trust_region_manifest.py"
require_file "build_spatial_holdout_validation_manifest.py"
require_file "summarize_spatial_refinement_round.py"
require_file "summarize_spatial_holdout_validation.py"
require_file "analyze_polarization_depth_progression.py"
require_file "make_corrected_ic_dense_inverse_posterior.py"
require_file "analyze_residual_sensitivity_hidden_physics.py"
require_file "generate_hidden_physics_report.py"

echo "========================================"
echo "BTO CORRECTED-IC POSTERIOR-GUIDED CAMPAIGN"
echo "========================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node list:     ${SLURM_NODELIST:-local}"
echo "Campaign dir:  $_script_dir"
echo "Prefix:        $PREFIX"
echo "Config:        $ACTIVE_CONFIG"
echo "Base input:    $BASE_INPUT"
echo "Anchor limit:  $ANCHOR_LIMIT"
echo "Refine rounds: $REFINE_ROUNDS"
echo "Holdout:       $HOLDOUT_POLICY"
echo "Post-analysis: $POST_ANALYSIS_DIR"
echo "Python:        $PYTHON"
"$PYTHON" --version
echo "MatEnsemble:   $MATENSEMBLE"
echo "========================================"

"$PYTHON" build_posterior_guided_corrected_ic_config.py \
  --campaign-dir "$_script_dir" \
  --base-config sim_bo_config_interactive.json \
  --output "$ACTIVE_CONFIG" \
  --max-candidates "$ANCHOR_LIMIT"

round_done() {
  local round_name="$1"
  local obs="$_script_dir/round_${round_name}/observations.json"
  "$PYTHON" - "$obs" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text())
if isinstance(data, dict):
    data = data.get("observations", [])
valid = [
    row for row in data
    if row.get("converged", True) and int(row.get("n_conditions", 0) or 0) > 0
]
raise SystemExit(0 if valid else 1)
PY
}

run_manifest_round() {
  local round_name="$1"
  local manifest="$_script_dir/round_${round_name}/manifest.json"
  if round_done "$round_name"; then
    echo "SKIPPING completed round: $round_name"
    return 0
  fi
  echo "========================================"
  echo "RUNNING ROUND: $round_name"
  echo "Manifest: $manifest"
  echo "========================================"
  "$MATENSEMBLE" run_manifest_round.py --manifest "$manifest"
}

save_best_env() {
  local observations="$1"
  local output_env="$2"
  local keys="$3"
  "$PYTHON" select_best_observation_params.py \
    --observations "$observations" \
    --keys "$keys" \
    --format shell > "$output_env"
  cat "$output_env"
}

anchor_round="${PREFIX}_anchor_000"
"$PYTHON" build_corrected_ic_anchor_manifest.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --round "$anchor_round" \
  --base-input "$BASE_INPUT"
run_manifest_round "$anchor_round"

anchor_best_env="$_script_dir/condition_gp_active/${anchor_round}_best_g.env"
save_best_env "$_script_dir/round_${anchor_round}/observations.json" "$anchor_best_env" "g11,g12,g44"
source "$anchor_best_env"
echo "Selected posterior-guided corrected-IC best g: g11=$G11 g12=$G12 g44=$G44"

dense_dir="$_script_dir/${POST_ANALYSIS_DIR}/inverse_posterior_dense"
"$PYTHON" make_corrected_ic_dense_inverse_posterior.py \
  --campaign-dir "$_script_dir" \
  --round "$anchor_round" \
  --output-dir "$dense_dir" \
  --num-samples 65536

"$PYTHON" analyze_residual_sensitivity_hidden_physics.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --observed "round_${anchor_round}/observations.json" \
  --observed-round "round_${anchor_round}" \
  --posterior-summary "${POST_ANALYSIS_DIR}/inverse_posterior_dense/posterior_summary.json" \
  --output-dir "${POST_ANALYSIS_DIR}/residual_sensitivity_anchor" \
  --best-round "round_${anchor_round}" \
  --best-candidate-id "$CANDIDATE_ID"

scalar_round="${PREFIX}_hidden_scalar_000"
"$PYTHON" build_hidden_physics_anchor_manifest.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --round "$scalar_round" \
  --base-input "$BASE_INPUT" \
  --g11 "$G11" \
  --g12 "$G12" \
  --g44 "$G44"
run_manifest_round "$scalar_round"

scalar_best_env="$_script_dir/condition_gp_active/${scalar_round}_best_hidden.env"
save_best_env "$_script_dir/round_${scalar_round}/observations.json" "$scalar_best_env" "g11,g12,g44,screen_lambda,vegard_strain"
source "$scalar_best_env"
echo "Selected scalar hidden base: screen_lambda=${SCREEN_LAMBDA:-NA} vegard_strain=${VEGARD_STRAIN:-NA}"

pilot_round="${PREFIX}_spatial_pilot_000"
"$PYTHON" build_spatial_hidden_physics_pilot_manifest.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --round "$pilot_round" \
  --base-input "$BASE_INPUT" \
  --use-best-scalar \
  --scalar-observations "$_script_dir/round_${scalar_round}/observations.json" \
  --g11 "$G11" \
  --g12 "$G12" \
  --g44 "$G44"
run_manifest_round "$pilot_round"

prev_round="$pilot_round"
ref0_round="${PREFIX}_spatial_refine_000"
"$PYTHON" build_spatial_refinement_trust_region_manifest.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --round "$ref0_round" \
  --base-input "$BASE_INPUT" \
  --center-observations "$_script_dir/round_${prev_round}/observations.json" \
  --candidate-design initial \
  --g11 "$G11" \
  --g12 "$G12" \
  --g44 "$G44"
run_manifest_round "$ref0_round"
"$PYTHON" summarize_spatial_refinement_round.py --campaign-dir "$_script_dir" --round "$ref0_round"
prev_round="$ref0_round"

for ((idx = 1; idx <= REFINE_ROUNDS; idx++)); do
  refine_round="$(printf '%s_spatial_refine_%03d' "$PREFIX" "$idx")"
  case "$idx" in
    1) step_scale="0.75" ;;
    2) step_scale="0.50" ;;
    *) step_scale="0.35" ;;
  esac
  "$PYTHON" build_spatial_refinement_trust_region_manifest.py \
    --campaign-dir "$_script_dir" \
    --config "$ACTIVE_CONFIG" \
    --round "$refine_round" \
    --base-input "$BASE_INPUT" \
    --center-observations "$_script_dir/round_${prev_round}/observations.json" \
    --candidate-design local \
    --step-scale "$step_scale" \
    --g11 "$G11" \
    --g12 "$G12" \
    --g44 "$G44"
  run_manifest_round "$refine_round"
  "$PYTHON" summarize_spatial_refinement_round.py --campaign-dir "$_script_dir" --round "$refine_round"
  prev_round="$refine_round"
done

holdout_round="${PREFIX}_holdout_full_000"
"$PYTHON" build_spatial_holdout_validation_manifest.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --round "$holdout_round" \
  --base-input "$BASE_INPUT" \
  --best-summary "$_script_dir/condition_gp_active/${prev_round}_best_summary.json" \
  --policy "$HOLDOUT_POLICY" \
  --g11 "$G11" \
  --g12 "$G12" \
  --g44 "$G44"
run_manifest_round "$holdout_round"
"$PYTHON" summarize_spatial_holdout_validation.py --campaign-dir "$_script_dir" --round "$holdout_round"

spatial_candidate_id="r${holdout_round}_cand02"
"$PYTHON" analyze_polarization_depth_progression.py \
  --campaign-dir "$_script_dir" \
  --round "$holdout_round" \
  --candidate-id "$spatial_candidate_id" \
  --pulse 0.43 \
  --threshold 0.05 \
  --output-dir "$_script_dir/${POST_ANALYSIS_DIR}/hidden_physics_report/${holdout_round}_pz_gt_0p05"

"$PYTHON" generate_hidden_physics_report.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --output-dir "${POST_ANALYSIS_DIR}/hidden_physics_report" \
  --scalar-round "$scalar_round" \
  --spatial-round "$pilot_round" \
  --refinement-summary "condition_gp_active/${ref0_round}_candidates.json"

echo "========================================"
echo "POSTERIOR-GUIDED CORRECTED-IC CAMPAIGN COMPLETE"
echo "Final refinement round: $prev_round"
echo "Final holdout round:    $holdout_round"
echo "Best g env:             $anchor_best_env"
echo "Dense posterior:        $dense_dir"
echo "Analysis bundle:        $_script_dir/$POST_ANALYSIS_DIR"
echo "========================================"
