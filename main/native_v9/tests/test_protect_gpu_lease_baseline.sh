#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dingxukai/TRACE/trace_role_bridge_native_v9
GUARD=${ROOT}/scripts/protect_gpu_lease.sh
MOCK_NVIDIA_SMI=${ROOT}/tests/mock_nvidia_smi_lease.sh
TEST_DIR=$(mktemp -d /tmp/trace_gpu_lease_baseline_test.XXXXXX)
RUN_ROOT=${TEST_DIR}/run
UUID_FILE=${TEST_DIR}/selected.tsv
BASELINE_FILE=${TEST_DIR}/baseline.tsv
FAKE_UUID=GPU-test-lease
PIPELINE_PID=
BASELINE_PID=
GUARD_PID=

cleanup() {
  [[ -z "${GUARD_PID}" ]] || kill "${GUARD_PID}" 2>/dev/null || true
  [[ -z "${PIPELINE_PID}" ]] || kill "${PIPELINE_PID}" 2>/dev/null || true
  [[ -z "${BASELINE_PID}" ]] || kill "${BASELINE_PID}" 2>/dev/null || true
  rm -rf "${TEST_DIR}"
}
trap cleanup EXIT

printf '0\t%s\n' "${FAKE_UUID}" > "${UUID_FILE}"
sleep 120 &
PIPELINE_PID=$!
sleep 120 &
BASELINE_PID=$!
BASELINE_IDENTITY=$(ps -o lstart= -p "${BASELINE_PID}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
printf '%s\t%s\t%s\n' "${FAKE_UUID}" "${BASELINE_PID}" "${BASELINE_IDENTITY}" > "${BASELINE_FILE}"

env FAKE_GPU_UUID="${FAKE_UUID}" FAKE_GPU_PID="${BASELINE_PID}" NVIDIA_SMI_BIN="${MOCK_NVIDIA_SMI}" GPU_LEASE_SCAN_SECONDS=0.1 "${GUARD}" "${RUN_ROOT}" "${PIPELINE_PID}" "${UUID_FILE}" "${BASELINE_FILE}" &
GUARD_PID=$!

sleep 1
if ! kill -0 "${BASELINE_PID}" 2>/dev/null; then
  echo "lease guard incorrectly terminated a launch-time baseline process" >&2
  exit 1
fi
if grep -q 'action=invader_detected' "${RUN_ROOT}.gpu_lease.log"; then
  echo "lease guard incorrectly classified a baseline process as an invader" >&2
  exit 1
fi

echo "gpu lease baseline allow-list test passed"
