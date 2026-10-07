#!/usr/bin/env bash
#SBATCH -A hpcl-mat269
#SBATCH -p hpcl-mat269
#SBATCH -q hpcl-mat269
#SBATCH -C BL
#SBATCH -J autopf_live_gate
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 00:20:00
#SBATCH -o /scratch/hpcl-mat269/sny/sim_bo_bto_physical_bounds/logs/%x-%j.out
#SBATCH -e /scratch/hpcl-mat269/sny/sim_bo_bto_physical_bounds/logs/%x-%j.err

set -euo pipefail
REPO=/scratch/hpcl-mat269/sny/AutoPF_workflow_integration
ROOT=/scratch/hpcl-mat269/sny/sim_bo_bto_physical_bounds
IMAGE=/scratch/hpcl-mat269/sny/containers/autopf-ferret-pathfinder.sif
TORCH_SITE=/scratch/hpcl-mat269/sny/python_envs/autopf-gp-torch260/lib/python3.12/site-packages
export PYTHONPATH=${REPO}:${ROOT}/pathfinder_setup:/scratch/hpcl-mat269/sny/vendor/MatEnsemble/src:${TORCH_SITE}:${PYTHONPATH:-}
export AUTOPF_SMOKE_OUTPUT=${ROOT}/pathfinder_runs/autopf_live_integration_release_gate_${SLURM_JOB_ID}
cd "${REPO}"
srun -N 1 -n 1 --ntasks-per-node=1 --cpus-per-task=128 \
  apptainer exec --env PYTHONPATH="${PYTHONPATH}" \
  --env AUTOPF_SMOKE_OUTPUT="${AUTOPF_SMOKE_OUTPUT}" \
  --bind /scratch/hpcl-mat269/sny:/scratch/hpcl-mat269/sny "${IMAGE}" \
  python3 examples/closed-loop-aecroscopywave/smoke_pathfinder_real_moose.py
