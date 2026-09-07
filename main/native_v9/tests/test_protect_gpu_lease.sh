#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dingxukai/TRACE/trace_role_bridge_native_v9
GUARD=${ROOT}/scripts/protect_gpu_lease.sh
MOCK_NVIDIA_SMI=${ROOT}/tests/mock_nvidia_smi_lease.sh
TEST_DIR=$(mktemp -d /tmp/trace_gpu_lease_test.XXXXXX)
RUN_ROOT=${TEST_DIR}/run
UUID_FILE=${TEST_DIR}/selected.tsv
BASELINE_FILE=${TEST_DIR}/baseline.tsv
FAKE_UUID=GPU-test-lease
PIPELINE_PID=
INVADER_PID=
GUARD_PID=

cleanup() {
  [[ -z "${GUARD_PID}" ]] || kill "${GUARD_PID}" 2>/dev/null || true
  [[ -z "${PIPELINE_PID}" ]] || kill "${PIPELINE_PID}" 2>/dev/null || true
  [[ -z "${INVADER_PID}" ]] || kill "${INVADER_PID}" 2>/dev/null || true
  rm -rf "${TEST_DIR}"
}
trap cleanup EXIT

printf '0\t%s\n' "${FAKE_UUID}" > "${UUID_FILE}"
: > "${BASELINE_FILE}"

sleep 120 &
PIPELINE_PID=$!
sleep 120 &
INVADER_PID=$!

FAKE_GPU_UUID=${FAKE_UUID} \
FAKE_GPU_PID=${INVADER_PID} \
NVIDIA_SMI_BIN=${MOCK_NVIDIA_SMI} \
GPU_LEASE_SCAN_SECONDS=0.1 \
GPU_LEASE_TERM_GRACE_SECONDS=1 \
  "${GUARD}" "${RUN_ROOT}" "${PIPELINE_PID}" "${UUID_FILE}" "${BASELINE_FILE}" &
GUARD_PID=$!

for _ in $(seq 1 50); do
  if ! kill -0 "${INVADER_PID}" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

if kill -0 "${INVADER_PID}" 2>/dev/null; then
  echo "lease guard failed to terminate the confirmed test invader" >&2
  exit 1
fi
if ! kill -0 "${PIPELINE_PID}" 2>/dev/null; then
  echo "lease guard incorrectly terminated the protected pipeline" >&2
  exit 1
fi
for _ in $(seq 1 50); do
  if grep -q 'action=terminate_complete' "${RUN_ROOT}.gpu_lease.log" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
grep -q 'action=invader_detected' "${RUN_ROOT}.gpu_lease.log"
grep -q 'action=terminate_complete' "${RUN_ROOT}.gpu_lease.log"

echo "gpu lease guard test passed"
