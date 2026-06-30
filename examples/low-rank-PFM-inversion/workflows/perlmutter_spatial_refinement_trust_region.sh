#!/bin/bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH --reservation _CAP_cnms_online_matEnsemble
#SBATCH -J bto_sprefine
#SBATCH -o logs/bto_sprefine_%j.out
#SBATCH -e logs/bto_sprefine_%j.err

set -euo pipefail

mkdir -p logs

export MPICH_GPU_SUPPORT_ENABLED=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-simbo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

module load python/3.11
source activate /global/cfs/cdirs/m5014/PhaseField/autopf_env

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_script_dir"

REFINE_ROUND="${REFINE_ROUND:-spatial_refine_trust_000}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-sim_bo_config_interactive.json}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"
CANDIDATE_DESIGN="${CANDIDATE_DESIGN:-initial}"
CENTER_OBSERVATIONS="${CENTER_OBSERVATIONS:-}"
TRUST_REGION_STEP_SCALE="${TRUST_REGION_STEP_SCALE:-1.0}"
RUN_REFINEMENT="${RUN_REFINEMENT:-0}"

echo "========================================"
echo "BTO SPATIAL HIDDEN-PHYSICS TRUST REGION"
echo "========================================"
echo "Job ID:          ${SLURM_JOB_ID:-interactive}"
echo "Node list:       ${SLURM_NODELIST:-local}"
echo "Campaign dir:    $_script_dir"
echo "Config:          $ACTIVE_CONFIG"
echo "Round:           $REFINE_ROUND"
echo "Candidate limit: $CANDIDATE_LIMIT"
echo "Candidate design:$CANDIDATE_DESIGN"
echo "Center obs:      ${CENTER_OBSERVATIONS:-<summary>}"
echo "Step scale:      $TRUST_REGION_STEP_SCALE"
echo "Run refinement:  $RUN_REFINEMENT"
echo "Python:          $(which python)"
python --version
echo "========================================"

limit_args=()
if [[ "$CANDIDATE_LIMIT" != "0" ]]; then
  limit_args=(--candidate-limit "$CANDIDATE_LIMIT")
fi
center_args=()
if [[ -n "$CENTER_OBSERVATIONS" ]]; then
  center_args=(--center-observations "$CENTER_OBSERVATIONS")
fi

python build_spatial_refinement_trust_region_manifest.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --round "$REFINE_ROUND" \
  --candidate-design "$CANDIDATE_DESIGN" \
  --step-scale "$TRUST_REGION_STEP_SCALE" \
  "${center_args[@]}" \
  "${limit_args[@]}"

python plot_spatial_vegard_field.py \
  --campaign-dir "$_script_dir"

if [[ "$RUN_REFINEMENT" == "1" ]]; then
  matensemble-launcher run_manifest_round.py \
    --manifest "$_script_dir/round_${REFINE_ROUND}/manifest.json"
else
  echo "Skipping MOOSE launch because RUN_REFINEMENT=$RUN_REFINEMENT"
  echo "To run inside an allocation: RUN_REFINEMENT=1 bash perlmutter_spatial_refinement_trust_region.sh"
fi
