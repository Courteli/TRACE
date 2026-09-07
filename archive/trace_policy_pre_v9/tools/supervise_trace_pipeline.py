#!/usr/bin/env python3
"""Wait for four GPUs, then run the GPU-only TRACE-Policy-v3 pipeline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime


ROOT = Path("/disk1/dingxukai/TRACE")
PROTECTED_SOURCES = (
    "run.py",
    "scripts/run_evidence.sh",
    "scripts/run_full_pipeline.sh",
    "scripts/run_stage0_cot.sh",
    "scripts/run_stage1_formation.sh",
    "scripts/run_stage2_refinement.sh",
    "src/configs/datasets/gsm8k_aug_nl.yaml",
    "src/configs/models/trace_policy_qwen3_instruct.yaml",
    "src/configs/trainer/trace_stage1_gpu4_dynamic.yaml",
    "src/datasets/gsm8k_aug_nl.py",
    "src/models/model_base.py",
    "src/models/read.py",
    "src/utils/distributed.py",
    "tools/isolated_gpu_ddp_entry.py",
    "src/models/trace_policy.py",
    "src/modules/trace_policy.py",
    "tools/trace_policy_ddp_memory_smoke.py",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def gpu_snapshot() -> list[dict]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, used, free, utilization = (
            int(value.strip()) for value in line.split(",")
        )
        rows.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "utilization_percent": utilization,
            }
        )
    return rows


def host_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024.0**2)
    raise RuntimeError("/proc/meminfo has no MemAvailable entry")


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} {message}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--stage0-checkpoint", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=3072)
    parser.add_argument("--minimum-host-available-gib", type=float, default=64.0)
    args = parser.parse_args()
    if not args.stage0_checkpoint.is_file():
        raise SystemExit(f"Missing Stage-0 checkpoint: {args.stage0_checkpoint}")

    output_dir = ROOT / "run_outputs" / "supervisor" / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    supervisor_log = output_dir / "supervisor.log"
    pipeline_log = output_dir / "pipeline.stdout.log"
    source_hashes = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in PROTECTED_SOURCES
    }
    write_json(output_dir / "source_hashes.json", source_hashes)

    status = {
        "created_at": now(),
        "model": "TRACE-Policy-v3",
        "run_tag": args.run_tag,
        "stage0_checkpoint": str(args.stage0_checkpoint),
        "state": "waiting_for_four_gpus",
        "required_stage1_gpus": 4,
        "stage1_effective_batch": 4,
        "stage1_gradient_accumulation_steps": 1,
        "stage1_cpu_activation_offload": False,
        "gpu_memory_threshold_mib": args.gpu_memory_threshold_mib,
        "minimum_host_available_gib": args.minimum_host_available_gib,
        "poll_seconds": args.poll_seconds,
        "supervisor_pid": os.getpid(),
    }
    write_json(status_path, status)
    append_log(
        supervisor_log,
        "Waiting for any four available GPUs; one qualifying snapshot launches GPU-only Stage 1.",
    )

    while True:
        snapshot = gpu_snapshot()
        available_gib = host_available_gib()
        idle_rows = sorted(
            (
                row
                for row in snapshot
                if row["memory_used_mib"]
                <= args.gpu_memory_threshold_mib
            ),
            key=lambda row: (row["memory_used_mib"], row["index"]),
        )
        status.update(
            {
                "updated_at": now(),
                "gpu_snapshot": snapshot,
                "host_available_gib": available_gib,
            }
        )
        if (
            len(idle_rows) >= 4
            and available_gib >= args.minimum_host_available_gib
        ):
            selected = [row["index"] for row in idle_rows[:4]]
            break
        write_json(status_path, status)
        time.sleep(args.poll_seconds)

    gpu_csv = ",".join(str(index) for index in selected)
    status.update(
        {
            "state": "pipeline_running",
            "stage1_physical_gpus": selected,
            "pipeline_started_at": now(),
            "pipeline_log": str(pipeline_log),
        }
    )
    append_log(
        supervisor_log,
        f"Four available GPUs detected once; launching GPU-only Stage 1 on {gpu_csv}.",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PIPELINE_TAG": args.run_tag,
            "STAGE0_CKPT_OVERRIDE": str(args.stage0_checkpoint),
            "TRAIN_SEED": "0",
        }
    )
    with pipeline_log.open("ab", buffering=0) as output:
        child = subprocess.Popen(
            [
                "bash",
                "scripts/run_full_pipeline.sh",
                gpu_csv,
                str(selected[0]),
            ],
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status["pipeline_pid"] = child.pid
        write_json(status_path, status)
        return_code = child.wait()

    status.update(
        {
            "state": "pipeline_completed" if return_code == 0 else "pipeline_failed",
            "pipeline_return_code": return_code,
            "pipeline_finished_at": now(),
            "updated_at": now(),
        }
    )
    write_json(status_path, status)
    append_log(
        supervisor_log,
        f"Formal pipeline exited with return code {return_code}.",
    )


if __name__ == "__main__":
    main()
