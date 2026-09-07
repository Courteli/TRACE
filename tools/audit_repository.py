#!/usr/bin/env python3
"""CPU-only snapshot audit. Does not load models, alter files, or launch jobs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{25,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{25,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    re.compile(rb"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
]
FORBIDDEN_SUFFIXES = {".ckpt", ".safetensors", ".pt", ".pth", ".pyc", ".pem", ".key", ".bin", ".log"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def audit(root: Path) -> dict:
    root = root.resolve(strict=True)
    issues = []
    files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts]
    for path in files:
        rel = path.relative_to(root)
        if path.is_symlink():
            issues.append(f"symlink: {rel}")
        if path.suffix in FORBIDDEN_SUFFIXES or path.name.startswith(".env"):
            issues.append(f"excluded artifact type: {rel}")
        if path.stat().st_size >= 50 * 1024 * 1024:
            issues.append(f"oversize file: {rel}")
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            issues.append(f"possible secret in {rel}")  # never print the value

    data_manifest = json.loads((root / "data/manifest.json").read_text())
    expected_data_paths = {entry["path"] for entry in data_manifest["files"]}
    actual_data_paths = {str(p.relative_to(root)) for p in (root / "data").rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual_data_paths != expected_data_paths:
        issues.append("data file inventory differs from manifest")
    for entry in data_manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            issues.append(f"data integrity mismatch: {entry['path']}")

    strict_val = root / "data/GSM8k-Aug-NL/readcot_qsa_qwen_dc/val.json"
    if not strict_val.is_file() or digest(strict_val) != "5f2ddd39f09f95d834a1b2840ce12d9fd2461bc909a755f1b8c429497bdfa49b":
        issues.append("native-v9 fixed validation fingerprint mismatch")
    model_manifest = json.loads((root / "models/base_reference/manifest.json").read_text())
    for entry in model_manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            issues.append(f"model reference integrity mismatch: {entry['path']}")

    source = json.loads((root / "docs/source_manifest.json").read_text())
    for entry in source["files"]:
        path = root / entry["path"]
        if not path.is_file() or digest(path) != entry["snapshot_sha256"]:
            issues.append(f"source integrity mismatch: {entry['path']}")
        if entry.get("core_method") and entry["source_sha256"] != entry["snapshot_sha256"]:
            issues.append(f"core method changed: {entry['path']}")

    archived_files = 0
    for entry in source["trees"]:
        tree_root = root / entry["path"]
        tree_files = sorted(p for p in tree_root.rglob("*") if p.is_file())
        tree_hasher = hashlib.sha256()
        for path in tree_files:
            tree_hasher.update((str(path.relative_to(tree_root)) + "\0" + digest(path) + "\n").encode())
        if len(tree_files) != entry["files"] or tree_hasher.hexdigest() != entry["snapshot_tree_sha256"]:
            issues.append(f"archive tree integrity mismatch: {entry['path']}")
        if entry["source_tree_sha256"] != entry["snapshot_tree_sha256"]:
            issues.append(f"archived source changed: {entry['path']}")
        archived_files += len(tree_files)

    python_files = list((root / "main/native_v9").rglob("*.py")) + list((root / "tools").glob("*.py"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as error:
            issues.append(f"Python syntax: {path.relative_to(root)}: {error}")
    shell_files = list((root / "main/native_v9/scripts").glob("*.sh")) + list((root / "scripts").glob("*.sh"))
    for path in shell_files:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode:
            issues.append(f"Bash syntax: {path.relative_to(root)}")
    result = {"files": len(files), "bytes": sum(p.stat().st_size for p in files), "data_files_verified": len(expected_data_paths), "model_reference_files_verified": len(model_manifest["files"]), "main_source_files_verified": len(source["files"]), "archived_source_files_verified": archived_files, "archive_trees_verified": len(source["trees"]), "main_python_syntax_checked": len(python_files), "shell_syntax_checked": len(shell_files), "issues": issues}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raise SystemExit(bool(audit(args.repo_root)["issues"]))


if __name__ == "__main__":
    main()
