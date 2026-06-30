#!/bin/bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 4
#SBATCH -J bto_resp_basis
#SBATCH -o logs/bto_resp_basis_%j.out
#SBATCH -e logs/bto_resp_basis_%j.err

set -euo pipefail

mkdir -p logs

export MPICH_GPU_SUPPORT_ENABLED=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-simbo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

module load python/3.11
source activate /global/cfs/cdirs/m5014/PhaseField/autopf_env

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_script_dir"

RESPONSE_ROUND="${RESPONSE_ROUND:-response_basis_compare_000}"
RESPONSE_POLICY="${RESPONSE_POLICY:-representative12}"
ACTIVE_CONFIG="${ACTIVE_CONFIG:-sim_bo_config_interactive.json}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-0}"
RUN_RESPONSE="${RUN_RESPONSE:-0}"
SCORE_ONLY="${SCORE_ONLY:-0}"

echo "========================================"
echo "BTO RESPONSE-BASIS COMPARISON"
echo "========================================"
echo "Job ID:          ${SLURM_JOB_ID:-interactive}"
echo "Node list:       ${SLURM_NODELIST:-local}"
echo "Campaign dir:    $_script_dir"
echo "Config:          $ACTIVE_CONFIG"
echo "Round:           $RESPONSE_ROUND"
echo "Policy:          $RESPONSE_POLICY"
echo "Candidate limit: $CANDIDATE_LIMIT"
echo "Run response:    $RUN_RESPONSE"
echo "Score only:      $SCORE_ONLY"
echo "Python:          $(which python)"
python --version
echo "========================================"

limit_args=()
if [[ "$CANDIDATE_LIMIT" != "0" ]]; then
  limit_args=(--candidate-limit "$CANDIDATE_LIMIT")
fi

python build_response_basis_comparison_manifest.py \
  --campaign-dir "$_script_dir" \
  --config "$ACTIVE_CONFIG" \
  --round "$RESPONSE_ROUND" \
  --policy "$RESPONSE_POLICY" \
  "${limit_args[@]}"

manifest="$_script_dir/round_${RESPONSE_ROUND}/manifest.json"

if [[ "$RUN_RESPONSE" == "1" ]]; then
  run_args=(--manifest "$manifest")
  if [[ "$SCORE_ONLY" == "1" ]]; then
    run_args+=(--score-only)
  fi
  matensemble-launcher run_manifest_round.py "${run_args[@]}"
  python summarize_response_basis_comparison.py \
    --campaign-dir "$_script_dir" \
    --round "$RESPONSE_ROUND"
else
  echo "Skipping MOOSE launch because RUN_RESPONSE=$RUN_RESPONSE"
  echo "To run inside an allocation: RUN_RESPONSE=1 bash perlmutter_response_basis_comparison.sh"
fi
