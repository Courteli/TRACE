#!/usr/bin/env python3
"""Read-only GPU contention observer. This program never sends signals."""

import csv
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time


def process(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        return int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


def descendant(pid, ancestor):
    for _ in range(64):
        if pid == ancestor:
            return True
        state = process(pid)
        if state is None or pid <= 1:
            return False
        pid = state[0]
    return False


def main():
    root, leader, uuid_file, _baseline_file = sys.argv[1:]
    leader = int(leader)
    identity = process(leader)
    if identity is None:
        return
    protected = {}
    for line in Path(uuid_file).read_text().splitlines():
        gpu, uuid = line.split("\t")
        protected[uuid] = gpu
    seen = set()
    with Path(root + ".gpu_observer.log").open("a", buffering=1) as out:
        def log(message):
            out.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {message}\n")
        log(f"observer_started leader={leader} policy=observe_only_never_kill")
        while (state := process(leader)) is not None and state[1] == identity[1]:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=8,
                )
                active = set()
                for uuid, raw_pid in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                    pid = int(raw_pid)
                    if uuid not in protected or descendant(pid, leader):
                        continue
                    other = process(pid)
                    if other is None:
                        continue
                    key = (uuid, pid, other[1])
                    active.add(key)
                    if key not in seen:
                        log(f"contention_detected gpu={protected[uuid]} pid={pid} action=log_only")
                seen = active
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                log(f"observation_error={error}")
            time.sleep(float(os.environ.get("GPU_LEASE_SCAN_SECONDS", "10")))
        log("observer_stopped")


if __name__ == "__main__":
    main()
