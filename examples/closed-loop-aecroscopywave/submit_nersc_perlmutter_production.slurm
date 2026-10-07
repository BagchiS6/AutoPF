#!/usr/bin/env bash
#SBATCH --account=m5014_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=autopf_live_nersc
#SBATCH --output=autopf_live_nersc-%j.out
#SBATCH --error=autopf_live_nersc-%j.err

set -euo pipefail

: "${SCRATCH:?NERSC must provide SCRATCH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AUTOPF_REPO="${AUTOPF_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export NERSC_DEPLOY_ROOT="${NERSC_DEPLOY_ROOT:-${SCRATCH}/autopf-aecroscopywave}"
export AUTOPF_ENV="${AUTOPF_ENV:-/global/cfs/cdirs/m5014/PhaseField/autopf_env}"
export MOOSE_APP="${MOOSE_APP:-/global/cfs/cdirs/m5014/PhaseField/MOOSE/projects/ferret/ferret-opt}"
export AUTOPF_EXPECTED_MOOSE_COMMIT="${AUTOPF_EXPECTED_MOOSE_COMMIT:-dd3d84a665}"
export AUTOPF_EXPECTED_APP_COMMIT="${AUTOPF_EXPECTED_APP_COMMIT:-c7809ccee6216ece36b7767c30ba0b2f5cb4f4f6}"
export AUTOPF_CONFIG="${AUTOPF_CONFIG:-${NERSC_DEPLOY_ROOT}/config/production.nersc-perlmutter.json}"
export AUTOPF_ITERATIONS="${AUTOPF_ITERATIONS:-1}"
export AUTOPF_OBJECTIVE_MODE="${AUTOPF_OBJECTIVE_MODE:-posterior_distance}"
export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0
export SLURM_CPU_BIND=cores
export MPLCONFIGDIR="${MPLCONFIGDIR:-${NERSC_DEPLOY_ROOT}/cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${NERSC_DEPLOY_ROOT}/cache}"
export PYTHONPATH="${AUTOPF_REPO}:${SCRIPT_DIR}:${PYTHONPATH:-}"

PYTHON="${AUTOPF_ENV}/bin/python"
for required in "${PYTHON}" "${MOOSE_APP}" "${AUTOPF_CONFIG}"; do
    if [[ ! -e "${required}" ]]; then
        echo "required NERSC deployment path is missing: ${required}" >&2
        exit 2
    fi
done
mkdir -p "${NERSC_DEPLOY_ROOT}/provenance" "${MPLCONFIGDIR}"

MOOSE_BANNER="$("${MOOSE_APP}" --version 2>&1)"
printf '%s\n' "${MOOSE_BANNER}" | tee "${NERSC_DEPLOY_ROOT}/provenance/moose_version_${SLURM_JOB_ID}.txt"
if [[ "${MOOSE_BANNER}" != *"${AUTOPF_EXPECTED_MOOSE_COMMIT}"* ]]; then
    echo "MOOSE provenance mismatch: expected ${AUTOPF_EXPECTED_MOOSE_COMMIT}" >&2
    exit 3
fi

APP_ROOT="$(dirname "${MOOSE_APP}")"
MOOSE_ROOT="$(cd "${APP_ROOT}/../.." && pwd)"
APP_COMMIT="$(git -C "${APP_ROOT}" rev-parse HEAD)"
MOOSE_COMMIT="$(git -C "${MOOSE_ROOT}" rev-parse HEAD)"
if [[ "${APP_COMMIT}" != "${AUTOPF_EXPECTED_APP_COMMIT}" ]]; then
    echo "MOOSE application provenance mismatch: expected ${AUTOPF_EXPECTED_APP_COMMIT}, found ${APP_COMMIT}" >&2
    exit 3
fi
if [[ "${MOOSE_COMMIT}" != "${AUTOPF_EXPECTED_MOOSE_COMMIT}"* ]]; then
    echo "MOOSE framework provenance mismatch: expected ${AUTOPF_EXPECTED_MOOSE_COMMIT}, found ${MOOSE_COMMIT}" >&2
    exit 3
fi
{
    printf 'site=NERSC Perlmutter\naccount=m5014_g\n'
    printf 'job_id=%s\napplication_commit=%s\nmoose_commit=%s\n' "${SLURM_JOB_ID}" "${APP_COMMIT}" "${MOOSE_COMMIT}"
    sha256sum "${MOOSE_APP}"
    stat "${MOOSE_APP}"
} > "${NERSC_DEPLOY_ROOT}/provenance/executable_${SLURM_JOB_ID}.txt"

"${PYTHON}" "${SCRIPT_DIR}/run_production.py" \
    --config "${AUTOPF_CONFIG}" \
    --objective-mode "${AUTOPF_OBJECTIVE_MODE}" \
    --preflight-only \
    --offline-instrument-preflight

if [[ "${AUTOPF_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

for required_variable in AFW_CLOUD_SERVER_URL AFW_INSTRUMENT_ID AFW_AGENT_SECRET; do
    if [[ -z "${!required_variable:-}" ]]; then
        echo "live run requires ${required_variable}" >&2
        exit 4
    fi
done

"${PYTHON}" "${SCRIPT_DIR}/run_production.py" \
    --config "${AUTOPF_CONFIG}" \
    --objective-mode "${AUTOPF_OBJECTIVE_MODE}" \
    --iterations "${AUTOPF_ITERATIONS}"
