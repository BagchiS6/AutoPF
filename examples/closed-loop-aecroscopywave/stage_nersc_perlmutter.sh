#!/usr/bin/env bash
set -euo pipefail

: "${SCRATCH:?NERSC sets SCRATCH; log in to Perlmutter before staging}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOPF_REPO="${AUTOPF_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NERSC_DEPLOY_ROOT="${NERSC_DEPLOY_ROOT:-${SCRATCH}/autopf-aecroscopywave}"
DEPLOY_SOURCE="${1:-}"

mkdir -p \
    "${NERSC_DEPLOY_ROOT}/calibration" \
    "${NERSC_DEPLOY_ROOT}/config" \
    "${NERSC_DEPLOY_ROOT}/inputs" \
    "${NERSC_DEPLOY_ROOT}/models" \
    "${NERSC_DEPLOY_ROOT}/moose" \
    "${NERSC_DEPLOY_ROOT}/state"

if [[ -n "${DEPLOY_SOURCE}" ]]; then
    for directory in calibration config inputs; do
        if [[ ! -d "${DEPLOY_SOURCE}/${directory}" ]]; then
            echo "missing deployment directory: ${DEPLOY_SOURCE}/${directory}" >&2
            exit 2
        fi
        rsync -a --checksum "${DEPLOY_SOURCE}/${directory}/" "${NERSC_DEPLOY_ROOT}/${directory}/"
    done
else
    cp -n "${SCRIPT_DIR}/physics_candidates.template.json" \
        "${NERSC_DEPLOY_ROOT}/config/physics_candidates.json"
    cp -n "${SCRIPT_DIR}/acquisition_plan.template.json" \
        "${NERSC_DEPLOY_ROOT}/config/acquisition_plan.json"
    echo "Templates staged. Add the reviewed calibration NPZ and MOOSE input before preflight." >&2
fi

cp "${SCRIPT_DIR}/production_config.nersc-perlmutter.template.json" \
    "${NERSC_DEPLOY_ROOT}/config/production.nersc-perlmutter.json"

printf 'AUTOPF_REPO=%s\nNERSC_DEPLOY_ROOT=%s\n' "${AUTOPF_REPO}" "${NERSC_DEPLOY_ROOT}"
