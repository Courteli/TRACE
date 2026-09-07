#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path


DATASET_ALIASES = {
    "GSM8K-Aug": "gsm8k_aug",
    "GSM8K": "gsm8k_aug",
    "GSM-Hard": "gsmhard",
    "GSMHard": "gsmhard",
    "SVAMP": "svamp",
    "MultiArith": "multiarith",
}

GEOMETRY_QUALITY_THRESHOLDS = {
    "min_n": 200,
    "min_n_views": 8.0,
    "min_diag_residual_cos": 0.0,
    "min_final_path_cos": 0.0,
    "min_step_norm_ratio": 0.05,
    "max_step_norm_ratio": 5.0,
    "min_view_signature_distance": 0.001,
    "min_assignment_entropy": 0.02,
    "max_assignment_entropy": 0.98,
    "min_assignment_progress_span": 0.05,
    "max_assignment_progress_inversion_frac": 0.35,
}


def read_manifest(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def parse_summary(path):
    text = path.read_text(errors="ignore")
    rows = {}
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line or "Label" in line:
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 8:
            continue
        dataset = DATASET_ALIASES.get(cols[1], cols[1])
        try:
            rows[dataset] = {
                "acc": float(cols[2]),
                "n_latent": float(cols[3]),
                "output_len": float(cols[4]),
                "L": float(cols[5]),
                "dep_f1": float(cols[6]),
                "res_sim": float(cols[7]),
            }
        except ValueError:
            continue
    avg_acc = None
    avg_l = None
    m = re.search(r"Average Acc:\s*([-0-9.]+)", text)
    if m:
        avg_acc = float(m.group(1))
    m = re.search(r"Average #L:\s*([-0-9.]+)", text)
    if m:
        avg_l = float(m.group(1))
    label = path.stem.removeprefix("summary_")
    return {"path": str(path), "label": label, "rows": rows, "avg_acc": avg_acc, "avg_L": avg_l}


def read_geometry(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {"path": str(path), "error": "could not parse json"}
    all_rows = data.get("all", {})
    return {
        "path": str(path),
        "n": all_rows.get("n", 0),
        "acc": all_rows.get("acc"),
        "L": all_rows.get("L"),
        "diag_residual_cos": all_rows.get("diag_residual_cos"),
        "final_path_cos": all_rows.get("final_path_cos"),
        "step_norm_ratio": all_rows.get("step_norm_ratio"),
        "view_signature_distance": all_rows.get("view_signature_distance"),
        "assignment_entropy": all_rows.get("assignment_entropy"),
        "assignment_progress_span": all_rows.get("assignment_progress_span"),
        "assignment_progress_inversion_frac": all_rows.get("assignment_progress_inversion_frac"),
        "assignment_progress_center_std": all_rows.get("assignment_progress_center_std"),
        "relation_density": all_rows.get("relation_density"),
        "n_views": all_rows.get("n_views"),
    }


def is_number(value):
    return isinstance(value, (int, float)) and value == value


def fmt_metric(value, digits=3):
    if not is_number(value):
        return "-"
    return f"{value:.{digits}f}"


def geometry_quality_checks(geometry):
    t = GEOMETRY_QUALITY_THRESHOLDS

    def ge(key, threshold):
        value = geometry.get(key)
        return is_number(value) and value >= threshold

    def le(key, threshold):
        value = geometry.get(key)
        return is_number(value) and value <= threshold

    return {
        "geometry_n_ge_200": ge("n", t["min_n"]),
        "geometry_n_views_ge_8": ge("n_views", t["min_n_views"]),
        "geometry_diag_residual_cos_ge_0": ge("diag_residual_cos", t["min_diag_residual_cos"]),
        "geometry_final_path_cos_ge_0": ge("final_path_cos", t["min_final_path_cos"]),
        "geometry_step_norm_ratio_ge_0_05": ge("step_norm_ratio", t["min_step_norm_ratio"]),
        "geometry_step_norm_ratio_le_5": le("step_norm_ratio", t["max_step_norm_ratio"]),
        "geometry_view_signature_distance_ge_0_001": ge(
            "view_signature_distance",
            t["min_view_signature_distance"],
        ),
        "geometry_assignment_entropy_ge_0_02": ge("assignment_entropy", t["min_assignment_entropy"]),
        "geometry_assignment_entropy_le_0_98": le("assignment_entropy", t["max_assignment_entropy"]),
        "geometry_assignment_progress_span_ge_0_05": ge(
            "assignment_progress_span",
            t["min_assignment_progress_span"],
        ),
        "geometry_assignment_progress_inversion_frac_le_0_35": le(
            "assignment_progress_inversion_frac",
            t["max_assignment_progress_inversion_frac"],
        ),
    }


def best_geometry_quality(geometries):
    if not geometries:
        return {}, {}
    scored = []
    for geometry in geometries:
        checks = geometry_quality_checks(geometry)
        score = sum(1 for value in checks.values() if value)
        scored.append((score, geometry.get("n", 0) or 0, geometry, checks))
    _, _, geometry, checks = max(scored, key=lambda item: (item[0], item[1]))
    return geometry, checks


def artifact_status_for_label(run_dir, label):
    visual_dir = run_dir / f"visual_{label}_gsm8k_aug"
    geometry_path = visual_dir / "geometry_200" / "trace_bridge_geometry_summary.json"
    visual_record = run_dir / f"eval_{label}_gsm8k_aug_logs" / "tb" / "run" / "trace_bridge_visual_test.pt"
    geometry = read_geometry(geometry_path) if geometry_path.exists() else {}
    geometry_quality = geometry_quality_checks(geometry) if geometry else {}
    return {
        "label": label,
        "visual_record": str(visual_record) if visual_record.exists() else None,
        "has_3d_paths_png": (visual_dir / "trace_bridge_paths_3d.png").exists(),
        "has_similarity_heatmap_png": (visual_dir / "trace_bridge_similarity_heatmap.png").exists(),
        "has_assignment_heatmap_png": (visual_dir / "trace_bridge_assignment_heatmap.png").exists(),
        "geometry": geometry,
        "geometry_quality": geometry_quality,
        "geometry_quality_pass": bool(geometry_quality) and all(geometry_quality.values()),
    }


def candidate_from_summary(run_dir, summary):
    label = summary["label"]
    row = summary["rows"].get("gsm8k_aug", {})
    artifacts = artifact_status_for_label(run_dir, label)
    return {
        "label": label,
        "summary_path": summary["path"],
        "gsm8k_acc": row.get("acc"),
        "avg_acc": summary.get("avg_acc"),
        "avg_L": summary.get("avg_L"),
        "artifacts": artifacts,
        "passes_metric": (
            is_number(row.get("acc"))
            and row.get("acc") >= 29.0
            and is_number(summary.get("avg_acc"))
            and summary.get("avg_acc") >= 29.0
        ),
        "passes_visual_geometry": (
            bool(artifacts["visual_record"])
            and artifacts["has_3d_paths_png"]
            and artifacts["has_similarity_heatmap_png"]
            and artifacts["has_assignment_heatmap_png"]
            and artifacts["geometry_quality_pass"]
        ),
    }


def choose_best_candidate(candidates):
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (
            bool(c["passes_metric"] and c["passes_visual_geometry"]),
            c["avg_acc"] if is_number(c.get("avg_acc")) else -1.0,
            c["gsm8k_acc"] if is_number(c.get("gsm8k_acc")) else -1.0,
            -(c["avg_L"] if is_number(c.get("avg_L")) else 1e9),
        ),
    )


def summarize_run(run_dir):
    manifest = read_manifest(run_dir / "manifest.txt")
    best_ckpts = sorted(str(p) for p in run_dir.glob("*best_ckpt.txt"))
    snapshot_ckpts = sorted(str(p) for p in run_dir.glob("ckpt_snapshots/*/checkpoints/*.ckpt"))
    summaries = [parse_summary(p) for p in sorted(run_dir.glob("summary_*.md"))]
    geometries = [read_geometry(p) for p in sorted(run_dir.glob("visual_*_gsm8k_aug/geometry_200/trace_bridge_geometry_summary.json"))]
    pngs = sorted(str(p) for p in run_dir.glob("visual_*_gsm8k_aug/*.png"))
    visual_records = sorted(str(p) for p in run_dir.glob("eval_*_gsm8k_aug_logs/tb/run/trace_bridge_visual_test.pt"))

    candidates = [candidate_from_summary(run_dir, summary) for summary in summaries]
    best_candidate = choose_best_candidate(candidates)
    best_gsm8k = best_candidate.get("gsm8k_acc") if best_candidate else None
    best_avg = best_candidate.get("avg_acc") if best_candidate else None
    best_l = best_candidate.get("avg_L") if best_candidate else None

    best_geometry, geometry_quality = best_geometry_quality(geometries)
    geometry_n = max((g.get("n", 0) or 0 for g in geometries), default=0)
    has_paths = any(p.endswith("trace_bridge_paths_3d.png") for p in pngs)
    has_similarity_heatmap = any(p.endswith("trace_bridge_similarity_heatmap.png") for p in pngs)
    has_assignment_heatmap = any(p.endswith("trace_bridge_assignment_heatmap.png") for p in pngs)
    candidate_ready = bool(
        best_candidate
        and best_candidate["passes_metric"]
        and best_candidate["passes_visual_geometry"]
    )

    checks = {
        "has_best_ckpt": bool(best_ckpts) or bool(snapshot_ckpts),
        "has_main_summary": bool(summaries),
        "same_label_metrics_and_artifacts_pass": candidate_ready,
        "gsm8k_acc_ge_29": best_gsm8k is not None and best_gsm8k >= 29.0,
        "avg_acc_ge_29": best_avg is not None and best_avg >= 29.0,
        "has_visual_record": bool(visual_records),
        "has_3d_paths_png": has_paths,
        "has_similarity_heatmap_png": has_similarity_heatmap,
        "has_assignment_heatmap_png": has_assignment_heatmap,
        "geometry_n_ge_200": geometry_n >= 200,
        "geometry_quality_pass": bool(geometry_quality) and all(geometry_quality.values()),
    }
    return {
        "run_dir": str(run_dir),
        "run_tag": run_dir.name,
        "manifest": manifest,
        "best_ckpts": best_ckpts,
        "snapshot_ckpts": snapshot_ckpts,
        "summaries": summaries,
        "geometries": geometries,
        "best_geometry": best_geometry,
        "geometry_quality": geometry_quality,
        "pngs": pngs,
        "visual_records": visual_records,
        "best_gsm8k_acc": best_gsm8k,
        "best_avg_acc": best_avg,
        "best_avg_L": best_l,
        "best_candidate": best_candidate,
        "candidates": candidates,
        "checks": checks,
    }


def write_markdown(report, out_path):
    lines = []
    lines.append("# TRACE-BRIDGE Result Audit")
    lines.append("")
    lines.append("| Run | Best Label | GSM8K Acc | Avg Acc | Avg #L | Checkpoint | Summary | Visual | Geometry | Quality | Gate |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |")
    for run in report["runs"]:
        c = run["checks"]
        gate = "ready" if all(c.values()) else "pending"
        visual = (
            "yes"
            if c["has_3d_paths_png"] and c["has_similarity_heatmap_png"] and c["has_assignment_heatmap_png"]
            else "no"
        )
        best_label = (run.get("best_candidate") or {}).get("label", "-")
        geometry = run.get("best_geometry") or {}
        geometry_n = max((g.get("n", 0) or 0 for g in run["geometries"]), default=0)
        geometry_cell = "-"
        if geometry:
            geometry_cell = (
                f"n={geometry.get('n', '-')}, views={geometry.get('n_views', '-')}, "
                f"final={fmt_metric(geometry.get('final_path_cos'))}, "
                f"view={fmt_metric(geometry.get('view_signature_distance'), 4)}, "
                f"prog={fmt_metric(geometry.get('assignment_progress_span'), 3)}"
            )
        lines.append(
            "| {run} | {label} | {gsm} | {avg} | {l} | {ckpt} | {summary} | {visual} | {geom} | {quality} | {gate} |".format(
                run=run["run_tag"],
                label=best_label,
                gsm=f"{run['best_gsm8k_acc']:.2f}" if run["best_gsm8k_acc"] is not None else "-",
                avg=f"{run['best_avg_acc']:.2f}" if run["best_avg_acc"] is not None else "-",
                l=f"{run['best_avg_L']:.2f}" if run["best_avg_L"] is not None else "-",
                ckpt="yes" if c["has_best_ckpt"] else "no",
                summary="yes" if c["has_main_summary"] else "no",
                visual=visual,
                geom=geometry_cell if geometry else str(geometry_n),
                quality="yes" if c["geometry_quality_pass"] else "no",
                gate=gate,
            )
        )
    lines.append("")
    lines.append(
        "Gate requires: best ckpt, and one same-label/test checkpoint with summary, GSM8K Acc >= 29, Avg Acc >= 29, visual record, 3D PNG, similarity heatmap PNG, assignment heatmap PNG, geometry n >= 200, and geometry quality sanity checks."
    )
    lines.append("")
    lines.append("Geometry quality checks:")
    lines.append("")
    for key, value in GEOMETRY_QUALITY_THRESHOLDS.items():
        lines.append(f"- {key}: {value}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge")
    parser.add_argument("--out", default="/disk1/dingxukai/trace_colar/run_outputs/trace_bridge/result_audit_20260707")
    parser.add_argument("--run-regex", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.txt").exists())
    if args.run_regex:
        pattern = re.compile(args.run_regex)
        run_dirs = [p for p in run_dirs if pattern.search(p.name)]
    report = {"root": str(root), "runs": [summarize_run(p) for p in run_dirs]}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "trace_bridge_result_audit.json"
    md_path = out_dir / "trace_bridge_result_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(md_path.read_text())
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
