#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dingxukai/TRACE/trace_role_bridge_native_v9
GUARD=${ROOT}/scripts/protect_gpu_lease.sh
MOCK_NVIDIA_SMI=${ROOT}/tests/mock_nvidia_smi_lease.sh
TEST_DIR=$(mktemp -d /tmp/trace_gpu_lease_descendant_test.XXXXXX)
RUN_ROOT=${TEST_DIR}/run
UUID_FILE=${TEST_DIR}/selected.tsv
BASELINE_FILE=${TEST_DIR}/baseline.tsv
CHILD_PID_FILE=${TEST_DIR}/child.pid
FAKE_UUID=GPU-test-lease
PIPELINE_PID=
CHILD_PID=
GUARD_PID=

cleanup() {
  [[ -z "${GUARD_PID}" ]] || kill "${GUARD_PID}" 2>/dev/null || true
  [[ -z "${PIPELINE_PID}" ]] || kill "${PIPELINE_PID}" 2>/dev/null || true
  [[ -z "${CHILD_PID}" ]] || kill "${CHILD_PID}" 2>/dev/null || true
  rm -rf "${TEST_DIR}"
}
trap cleanup EXIT

printf '0\t%s\n' "${FAKE_UUID}" > "${UUID_FILE}"
: > "${BASELINE_FILE}"

bash -c 'sleep 120 & child=$!; printf "%s\n" "$child" > "$1"; wait "$child"' _ "${CHILD_PID_FILE}" &
PIPELINE_PID=$!
for _ in $(seq 1 50); do
  [[ -s "${CHILD_PID_FILE}" ]] && break
  sleep 0.1
done
CHILD_PID=$(<"${CHILD_PID_FILE}")

env FAKE_GPU_UUID="${FAKE_UUID}" FAKE_GPU_PID="${CHILD_PID}" NVIDIA_SMI_BIN="${MOCK_NVIDIA_SMI}" GPU_LEASE_SCAN_SECONDS=0.1 "${GUARD}" "${RUN_ROOT}" "${PIPELINE_PID}" "${UUID_FILE}" "${BASELINE_FILE}" &
GUARD_PID=$!

sleep 1
if ! kill -0 "${CHILD_PID}" 2>/dev/null; then
  echo "lease guard incorrectly terminated a pipeline descendant" >&2
  exit 1
fi
if grep -q 'action=invader_detected' "${RUN_ROOT}.gpu_lease.log"; then
  echo "lease guard incorrectly classified a pipeline descendant as an invader" >&2
  exit 1
fi

echo "gpu lease descendant allow-list test passed"
