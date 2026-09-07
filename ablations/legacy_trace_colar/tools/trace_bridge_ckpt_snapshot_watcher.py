#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def tmux_sessions():
    proc = subprocess.run(["tmux", "ls"], check=False, text=True, capture_output=True)
    sessions = set()
    for line in proc.stdout.splitlines():
        if ":" in line:
            sessions.add(line.split(":", 1)[0])
    return sessions


def stable_file(path: Path, min_age_seconds: int):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if stat.st_size <= 0:
        return None
    if time.time() - stat.st_mtime < min_age_seconds:
        return None
    return stat


def run_tag_and_phase(run_dir: Path, marker: str):
    name = run_dir.name
    if marker not in name:
        return None, None
    tail = marker + name.split(marker, 1)[1]
    for phase in ("stage2_conservative", "stage2_strong", "stage1", "bridge_baseline"):
        suffix = f"_{phase}"
        if tail.endswith(suffix):
            return tail[: -len(suffix)], phase
    return tail, "unknown"


def load_state(path: Path):
    if not path.exists():
        return {"copied": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"copied": {}}


def save_state(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def checkpoint_snapshot_name(ckpt: Path, stat):
    stem = ckpt.stem.replace("/", "_")
    if ckpt.name == "last.ckpt":
        return f"last__mtime{stat.st_mtime_ns}__size{stat.st_size}.ckpt"
    return f"{stem}__mtime{stat.st_mtime_ns}__size{stat.st_size}.ckpt"


def copy_one(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def prune_snapshot_dir(snapshot_ckpt_dir: Path, max_snapshots: int):
    if max_snapshots <= 0 or not snapshot_ckpt_dir.exists():
        return 0
    ckpts = sorted(
        snapshot_ckpt_dir.glob("*.ckpt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    pruned = 0
    for old in ckpts[max_snapshots:]:
        try:
            old.unlink()
            pruned += 1
        except FileNotFoundError:
            pass
    return pruned


def discover_run_dirs(log_root: Path, marker: str):
    roots = []
    roots.extend(log_root.glob(f"trace_bridge_qwen3_instruct*/qsa-gsm/*{marker}*"))
    roots.extend(log_root.glob(f"bridge_qwen3_instruct_hybrid_compact_anchor_gate/qsa-gsm/*{marker}*"))
    return sorted(p for p in roots if p.is_dir() and (p / "hparams.yaml").exists())


def snapshot_once(args):
    out_root = Path(args.out_root)
    state_path = Path(args.state)
    state = load_state(state_path)
    manifest_rows = []
    copied_now = 0

    for run_dir in discover_run_dirs(Path(args.log_root), args.run_marker):
        run_tag, phase = run_tag_and_phase(run_dir, args.run_marker)
        if not run_tag:
            continue
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        snapshot_run_dir = out_root / run_tag / "ckpt_snapshots" / run_dir.name
        snapshot_ckpt_dir = snapshot_run_dir / "checkpoints"
        hparams_src = run_dir / "hparams.yaml"
        hparams_dst = snapshot_run_dir / "hparams.yaml"
        if hparams_src.exists() and (not hparams_dst.exists() or hparams_src.stat().st_mtime_ns != hparams_dst.stat().st_mtime_ns):
            copy_one(hparams_src, hparams_dst)

        for ckpt in sorted(ckpt_dir.glob("*.ckpt")):
            stat = stable_file(ckpt, args.min_age_seconds)
            if stat is None:
                continue
            key = f"{ckpt}::{stat.st_mtime_ns}::{stat.st_size}"
            if key in state["copied"]:
                continue
            dst = snapshot_ckpt_dir / checkpoint_snapshot_name(ckpt, stat)
            if not dst.exists():
                copy_one(ckpt, dst)
                copied_now += 1
            row = {
                "copied_at": datetime.now().strftime("%F %T"),
                "run_tag": run_tag,
                "phase": phase,
                "source_run_dir": str(run_dir),
                "source_ckpt": str(ckpt),
                "snapshot_ckpt": str(dst),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            state["copied"][key] = row
            manifest_rows.append(row)
        pruned = prune_snapshot_dir(snapshot_ckpt_dir, args.max_snapshots_per_source)
        if pruned:
            print(f"[snapshot] pruned={pruned} source={run_dir.name}", flush=True)

    if manifest_rows:
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_state(state_path, state)
    return copied_now


def watched_sessions_alive(watch_sessions):
    if not watch_sessions:
        return True
    sessions = tmux_sessions()
    return any(session in sessions for session in watch_sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default="/disk1/dingxukai/trace_colar/logs")
    parser.add_argument("--out-root", default="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge")
    parser.add_argument("--run-marker", default="20260707_trace_bridge_final_full_")
    parser.add_argument("--state", default="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_final_full_20260707/ckpt_snapshot_state.json")
    parser.add_argument("--manifest", default="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_final_full_20260707/ckpt_snapshots.jsonl")
    parser.add_argument("--min-age-seconds", type=int, default=120)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-snapshots-per-source", type=int, default=24)
    parser.add_argument("--watch-sessions", nargs="*", default=[])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    copied = snapshot_once(args)
    print(f"[snapshot] copied={copied} at {datetime.now().strftime('%F %T')}", flush=True)
    if args.once:
        return

    while watched_sessions_alive(args.watch_sessions):
        time.sleep(args.interval_seconds)
        copied = snapshot_once(args)
        print(f"[snapshot] copied={copied} at {datetime.now().strftime('%F %T')}", flush=True)

    copied = snapshot_once(args)
    print(f"[snapshot] final copied={copied} at {datetime.now().strftime('%F %T')}", flush=True)


if __name__ == "__main__":
    main()
