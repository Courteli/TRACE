#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/disk1/dingxukai/trace_colar
OUT=${ROOT}/run_outputs/trace/20260716_stage1_component_ablations_gpu0
RUN_ROOT=${ROOT}/run_roots/trace/20260716_stage1_component_ablations_gpu0
LOG_ROOT=${ROOT}/logs/trace_bridge_qwen3_instruct_vizstrong/qsa-gsm
PARENT_PID=${1:?parent pipeline PID is required}

printf '%s\n' "$$" > "${OUT}/log_link_supervisor.pid"

while kill -0 "${PARENT_PID}" 2>/dev/null; do
  for variant in no_path_consistency no_progress_anchor no_multiview; do
    expected=${OUT}/${variant}/train_logs/tb/run
    [[ -e "${expected}" ]] && continue
    actual="$(
      for candidate in "${LOG_ROOT}"/20*; do
        [[ -f "${candidate}/hparams.yaml" ]] || continue
        if rg -q -F "${RUN_ROOT}/${variant}/train" "${candidate}/hparams.yaml"; then
          printf '%s\n' "${candidate}"
        fi
      done | sort | tail -n 1
    )"
    if [[ -n "${actual}" ]]; then
      mkdir -p "$(dirname "${expected}")"
      ln -s "${actual}" "${expected}"
      printf '%s [linked] variant=%s logger=%s\n' "$(date '+%F %T')" "${variant}" "${actual}" \
        >> "${OUT}/log_link_supervisor.log"
    fi
    for dataset in gsm8k gsmhard svamp multiarith; do
      eval_expected=${OUT}/${variant}/eval/${dataset}/logs/tb/run
      [[ -e "${eval_expected}" ]] && continue
      eval_actual="$(
        for candidate in "${LOG_ROOT}"/20*; do
          [[ -f "${candidate}/hparams.yaml" ]] || continue
          if rg -q -F "${RUN_ROOT}/${variant}/eval/${dataset}" "${candidate}/hparams.yaml"; then
            printf '%s\n' "${candidate}"
          fi
        done | sort | tail -n 1
      )"
      if [[ -n "${eval_actual}" ]]; then
        mkdir -p "$(dirname "${eval_expected}")"
        ln -s "${eval_actual}" "${eval_expected}"
        printf '%s [linked-eval] variant=%s dataset=%s logger=%s\n' \
          "$(date '+%F %T')" "${variant}" "${dataset}" "${eval_actual}" \
          >> "${OUT}/log_link_supervisor.log"
      fi
    done
  done
  sleep 60
done

printf '%s [done] parent pipeline exited\n' "$(date '+%F %T')" >> "${OUT}/log_link_supervisor.log"
