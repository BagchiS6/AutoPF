#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 9 ]]; then
    echo "usage: $0 WORKDIR SIMULATION_ID ITERATION METADATA_JSON PARAMETERS_JSON MOOSE_APP -i INPUT [OVERRIDES...]" >&2
    exit 2
fi

workdir="$1"
simulation_id="$2"
iteration="$3"
metadata_json="$4"
parameters_json="$5"
shift 5
rank="${FLUX_TASK_RANK:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
mkdir -p "${workdir}"
done_marker="${workdir}/rank0_postprocess.done"
start_epoch="$(date +%s.%N)"

if [[ "${rank}" == "0" ]]; then
    rm -f "${done_marker}"
    printf '%s\n' "${start_epoch}" > "${workdir}/native_start_epoch.txt"
fi

cd "${workdir}"
"$@"
moose_rc=$?

if [[ "${rank}" == "0" ]]; then
    date +%s.%N > "${workdir}/native_end_epoch.txt"
    printf '%s\n' "${moose_rc}" > "${workdir}/native_returncode.txt"
    "${AUTOPF_PYTHON}" "${AUTOPF_BTO_FINALIZER}" \
        --run-dir "${workdir}" \
        --simulation-id "${simulation_id}" \
        --iteration "${iteration}" \
        --metadata-json "${metadata_json}" \
        --parameters-json "${parameters_json}" \
        --start-epoch "${start_epoch}" \
        --moose-returncode "${moose_rc}" \
        --output "${workdir}/result.json" \
        --pickle-output "${workdir}/result.pickle"
    finalizer_rc=$?
    touch "${done_marker}"
    exit "${finalizer_rc}"
fi

for _ in $(seq 1 900); do
    [[ -f "${done_marker}" ]] && exit 0
    sleep 1
done
echo "timed out waiting for rank-0 MOOSE postprocessing: ${done_marker}" >&2
exit 124
