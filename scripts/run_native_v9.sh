#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if (( $# > 1 )); then echo 'Usage: run_native_v9.sh [--dry-run] [new-run-directory]' >&2; exit 2; fi
: "${TRACE_MODEL_PATH:?Set TRACE_MODEL_PATH to the complete original base-model directory}"
: "${PHYSICAL_GPUS:?Set PHYSICAL_GPUS to four GPUs allocated to your task}"
IFS=',' read -r -a GPU_IDS <<< "${PHYSICAL_GPUS}"
[[ ${#GPU_IDS[@]} -eq 4 ]] || { echo 'Exactly four GPU indices are required.' >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
  [[ $gpu =~ ^[0-9]+$ && ! ${SEEN_GPUS[$gpu]+set} ]] || { echo 'GPU indices must be distinct integers.' >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
done
export TRACE_SOURCE_ROOT=${REPO_ROOT}/main/native_v9
export TRACE_DATA_ROOT=${TRACE_DATA_ROOT:-${REPO_ROOT}/data}
export TRACE_PYTHON=${TRACE_PYTHON:-python}
RUN_ROOT=${1:-${REPO_ROOT}/runs/native_v9_$(date +%Y%m%d-%H%M%S)}
"${TRACE_PYTHON}" -B "${REPO_ROOT}/tools/audit_repository.py" --repo-root "${REPO_ROOT}"
if (( DRY_RUN )); then
  printf 'Dry run only; no GPU allocation or training. Command:\n'
  printf 'TRACE_MODEL_PATH=%q TRACE_DATA_ROOT=%q PHYSICAL_GPUS=%q bash %q %q\n' "${TRACE_MODEL_PATH}" "${TRACE_DATA_ROOT}" "${PHYSICAL_GPUS}" "${TRACE_SOURCE_ROOT}/scripts/run_full_native_v9.sh" "${RUN_ROOT}"
  exit 0
fi
"${TRACE_PYTHON}" -B -c 'import json,os,pathlib; p=pathlib.Path(os.environ["TRACE_MODEL_PATH"]); i=json.loads((p/"model.safetensors.index.json").read_text()); missing=[s for s in set(i["weight_map"].values()) if not (p/s).is_file()]; assert not missing, f"Missing model shards: {missing}"'
exec bash "${TRACE_SOURCE_ROOT}/scripts/run_full_native_v9.sh" "${RUN_ROOT}"
