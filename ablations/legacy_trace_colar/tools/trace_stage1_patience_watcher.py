#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


def latest_run(log_root: Path, run_contains: str) -> Path:
    runs = [p for p in log_root.glob("*") if p.is_dir() and run_contains in p.name]
    if not runs:
        raise FileNotFoundError(f"no run matching {run_contains!r} under {log_root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def monitor_score(path: Path):
    match = re.search(r"monitor(-?\d+(?:\.\d+)?)", path.name)
    return float(match.group(1)) if match else None


def best_checkpoint(log_root: Path, run_contains: str) -> Path:
    scored = []
    lasts = []
    for run_dir in log_root.glob("*"):
        if not run_dir.is_dir() or run_contains not in run_dir.name:
            continue
        for ckpt in run_dir.glob("checkpoints/*.ckpt"):
            if ckpt.name == "last.ckpt":
                lasts.append(ckpt)
                continue
            score = monitor_score(ckpt)
            if score is not None:
                scored.append((score, ckpt.stat().st_mtime, ckpt))
    if scored:
        return max(scored, key=lambda item: (item[0], item[1]))[2]
    if lasts:
        return max(lasts, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"no checkpoint matching {run_contains!r} under {log_root}")


def read_monitor_points(run_dir: Path):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        raise RuntimeError(f"tensorboard is required: {exc}") from exc

    events = sorted(run_dir.glob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    tag_points = {}
    for event in events:
        try:
            ea = event_accumulator.EventAccumulator(str(event), size_guidance={"scalars": 0})
            ea.Reload()
        except Exception:
            continue
        tags = ea.Tags().get("scalars", [])
        for tag in ("monitor", "val/monitor", "val/acc"):
            if tag not in tags:
                continue
            vals = ea.Scalars(tag)
            if vals:
                tag_points.setdefault(tag, []).extend(vals)

    for tag in ("monitor", "val/monitor", "val/acc"):
        vals = tag_points.get(tag, [])
        if not vals:
            continue
        deduped = {}
        for item in vals:
            deduped[(item.step, round(float(item.value), 10))] = item
        points = sorted(deduped.values(), key=lambda x: (x.step, x.wall_time))
        return tag, [
            {"step": int(p.step), "value": float(p.value), "wall_time": float(p.wall_time)}
            for p in points
        ]
    return "", []


def patience_state(points, min_delta: float):
    best = None
    best_idx = -1
    bad_count = 0
    history = []
    for idx, point in enumerate(points):
        value = point["value"]
        improved = best is None or value > best + min_delta
        if improved:
            best = value
            best_idx = idx
            bad_count = 0
        else:
            bad_count += 1
        item = dict(point)
        item["index"] = idx
        item["improved"] = improved
        item["best_so_far"] = best
        item["bad_count"] = bad_count
        history.append(item)
    return {
        "best": best,
        "best_index": best_idx,
        "bad_count": bad_count,
        "history": history,
    }


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(pid: int, timeout_sec: int = 90):
    if not pid or not process_alive(pid):
        return {"pid": pid, "terminated": False, "reason": "not_alive"}
    pgid = os.getpgid(pid)
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not process_alive(pid):
            return {"pid": pid, "pgid": pgid, "terminated": True, "signal": "SIGTERM"}
        time.sleep(1)
    os.killpg(pgid, signal.SIGKILL)
    return {"pid": pid, "pgid": pgid, "terminated": True, "signal": "SIGKILL"}


def tmux_session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def start_continuation(args, ckpt: Path):
    if tmux_session_exists(args.continue_session):
        return {"started": False, "reason": "tmux_session_exists", "session": args.continue_session}

    env_parts = {
        "RUN_NAME": args.run_name,
        "ARTIFACT_DIR": str(args.artifact_dir),
        "STAGE1_CKPT": str(ckpt),
        "STAGE2_GPU": str(args.stage2_gpu),
        "TEST_TIMES": str(args.test_times),
        "MAX_L": str(args.max_l),
        "MIN_L": str(args.min_l),
        "RUN_COLAR_BASELINE": args.run_colar_baseline,
    }
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_parts.items())
    command = f"cd {shlex.quote(str(args.root))} && {prefix} bash {shlex.quote(args.continue_script)}"
    subprocess.run(["tmux", "new-session", "-d", "-s", args.continue_session, command], check=True)
    return {"started": True, "session": args.continue_session, "command": command}


def write_status(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/disk1/dingxukai/trace_colar")
    parser.add_argument("--log-root", default="logs/trace_multipath_qwen3_instruct/gsm8k_aug_nl-gsm8k_aug_nl")
    parser.add_argument("--run-contains", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--kill-pid", type=int, default=0)
    parser.add_argument("--continue-session", required=True)
    parser.add_argument("--continue-script", default="run_trace_multipath_v3_three_stage_continue_from_stage1_20260706.sh")
    parser.add_argument("--stage2-gpu", default="5")
    parser.add_argument("--test-times", type=int, default=5)
    parser.add_argument("--max-l", type=int, default=40)
    parser.add_argument("--min-l", type=int, default=0)
    parser.add_argument("--run-colar-baseline", choices=["True", "False"], default="False")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.root = Path(args.root)
    args.log_root = args.root / args.log_root
    args.artifact_dir = Path(args.artifact_dir)
    status_path = args.artifact_dir / "stage1_patience4_status.json"
    decision_path = args.artifact_dir / "stage1_patience4_decision.json"

    while True:
        now = datetime.now().strftime("%F %T")
        try:
            run_dir = latest_run(args.log_root, args.run_contains)
            tag, points = read_monitor_points(run_dir)
            state = patience_state(points, args.min_delta)
            payload = {
                "time": now,
                "run_dir": str(run_dir),
                "tag": tag,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "n_validation_checks": len(points),
                "best": state["best"],
                "best_index": state["best_index"],
                "bad_count": state["bad_count"],
                "triggered": bool(len(points) > 0 and state["bad_count"] >= args.patience),
                "history": state["history"],
            }
            write_status(status_path, payload)
            print(
                f"[{now}] tag={tag or '-'} n={len(points)} best={state['best']} "
                f"bad_count={state['bad_count']} triggered={payload['triggered']}",
                flush=True,
            )
            if payload["triggered"]:
                ckpt = best_checkpoint(args.log_root, args.run_contains)
                payload["stage1_best_ckpt"] = str(ckpt)
                if args.dry_run:
                    payload["dry_run"] = True
                    write_status(decision_path, payload)
                    return
                payload["termination"] = terminate_process_group(args.kill_pid) if args.kill_pid else {}
                payload["continuation"] = start_continuation(args, ckpt)
                write_status(decision_path, payload)
                return
        except Exception as exc:
            payload = {"time": now, "error": repr(exc), "triggered": False}
            write_status(status_path, payload)
            print(f"[{now}] watcher error: {exc}", flush=True)

        if args.once:
            return
        if args.kill_pid and not process_alive(args.kill_pid):
            print(f"[{now}] kill_pid={args.kill_pid} is no longer alive; watcher exits", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
