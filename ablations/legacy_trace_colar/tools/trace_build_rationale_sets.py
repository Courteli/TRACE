#!/usr/bin/env python3
"""Generate, verify, deduplicate, and merge TRACE multi-rationale teachers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


PROMPT_VERSION = "trace-rationale-v1"
DEFAULT_SOURCE = Path(
    "/home/dingxukai/RoT/data/GSM8k-Aug-NL/readcot_qsa_qwen_dc"
)
DEFAULT_MODEL = Path(
    "/home/dingxukai/RoT/ckpt/base/Qwen3-4B-Instruct/"
    "qwen3-instruct"
)
NUMBER_PATTERN = r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?"
ANSWER_PATTERNS = (
    re.compile(rf"(?:final\s+)?answer\s*[:=]\s*({NUMBER_PATTERN})", re.I),
    re.compile(rf"####\s*({NUMBER_PATTERN})", re.I),
)
EQUATION_RESULT_PATTERN = re.compile(
    rf"=\s*(?P<rhs>{NUMBER_PATTERN})"
)
STEP_PREFIX = re.compile(
    r"^\s*(?:step\s*)?(?:\d+|[a-z])[\s.):\-]+\s*",
    re.I,
)


def _number(value: str) -> Optional[Fraction]:
    text = str(value).strip().replace("$", "").replace(",", "")
    if text.endswith("."):
        text = text[:-1]
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def answers_match(left: str, right: str) -> bool:
    left_value = _number(left)
    right_value = _number(right)
    if left_value is not None and right_value is not None:
        scale = max(1.0, abs(float(right_value)))
        return abs(float(left_value - right_value)) <= 1e-6 * scale
    return str(left).strip().lower() == str(right).strip().lower()


def extract_answer(text: str) -> Optional[str]:
    matches = []
    for pattern in ANSWER_PATTERNS:
        matches.extend(pattern.findall(text))
    return matches[-1] if matches else None


def _eval_expression_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_expression_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(Decimal(str(node.value)))
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.UAdd, ast.USub),
    ):
        value = _eval_expression_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_expression_node(node.left)
        right = _eval_expression_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"Unsupported arithmetic node: {type(node).__name__}")


def evaluate_expression(expression: str) -> Optional[Fraction]:
    normalized = normalize_math_text(expression)
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    normalized = normalized.strip()
    if not normalized or not re.search(r"[+\-*/]", normalized):
        return None
    try:
        parsed = ast.parse(normalized, mode="eval")
        return _eval_expression_node(parsed)
    except (SyntaxError, ValueError, ZeroDivisionError, InvalidOperation):
        return None


def normalize_math_text(text: str) -> str:
    normalized = str(text)
    fraction_pattern = re.compile(
        r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
    )
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = fraction_pattern.sub(r"(\1)/(\2)", normalized)
    replacements = {
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/",
        r"\left": "",
        r"\right": "",
        "×": "*",
        "÷": "/",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    return normalized


def _arithmetic_tokens(
    text: str,
    expected: Optional[Fraction] = None,
) -> str:
    normalized = normalize_math_text(text)
    tokens = re.findall(r"\d+(?:\.\d+)?|[+\-*/()]", normalized)
    candidates = []
    for start, token in enumerate(tokens):
        if not re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        expression = "".join(tokens[start:])
        previous = None
        while expression != previous:
            previous = expression
            expression = expression.replace("()", "")
        if not re.search(r"[+\-*/]", expression):
            continue
        value = evaluate_expression(expression)
        if value is None:
            continue
        candidates.append((expression, value))
    if expected is not None:
        for expression, value in reversed(candidates):
            scale = max(1.0, abs(float(expected)))
            if abs(float(value - expected)) <= 1e-6 * scale:
                return expression
    return candidates[0][0] if candidates else "".join(tokens)


def _operation_tree(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _operation_tree(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return ("n", str(node.value))
    if isinstance(node, ast.UnaryOp):
        return (
            "neg" if isinstance(node.op, ast.USub) else "pos",
            _operation_tree(node.operand),
        )
    if isinstance(node, ast.BinOp):
        names = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
        }
        op_name = names.get(type(node.op))
        if op_name is None:
            raise ValueError("unsupported operator")
        return (op_name, _operation_tree(node.left), _operation_tree(node.right))
    raise ValueError("unsupported node")


def extract_equations(text: str) -> List[dict]:
    equations = []
    previous_result_end = 0
    for match in EQUATION_RESULT_PATTERN.finditer(text):
        prefix = text[previous_result_end : match.start()]
        boundary = max(
            prefix.rfind("\n"),
            prefix.rfind(";"),
            prefix.rfind(":"),
        )
        lhs_source = prefix[boundary + 1 :]
        rhs = match.group("rhs").strip()
        rhs_value = _number(rhs)
        lhs = _arithmetic_tokens(lhs_source, expected=rhs_value)
        lhs_value = evaluate_expression(lhs)
        valid = False
        if lhs_value is not None and rhs_value is not None:
            scale = max(1.0, abs(float(rhs_value)))
            valid = abs(float(lhs_value - rhs_value)) <= 1e-6 * scale
        try:
            tree = _operation_tree(ast.parse(lhs, mode="eval"))
        except (SyntaxError, ValueError):
            tree = ("unparsed", lhs)
        equations.append(
            {
                "lhs": lhs,
                "rhs": rhs,
                "valid": valid,
                "tree": tree,
            }
        )
        previous_result_end = match.end()
    return equations


def split_steps(text: str) -> List[str]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    clean = re.sub(r"</?think>", "", clean, flags=re.I)
    clean = re.split(r"(?:final\s+)?answer\s*:", clean, flags=re.I)[0]
    lines = []
    for raw_line in clean.splitlines():
        line = STEP_PREFIX.sub("", raw_line).strip(" -*\t")
        if line:
            lines.append(line)
    if len(lines) <= 1:
        sentence_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)
        lines = [
            STEP_PREFIX.sub("", part).strip(" -*\t")
            for part in sentence_parts
            if STEP_PREFIX.sub("", part).strip(" -*\t")
        ]
    return lines[:8]


def rationale_fingerprint(steps: Sequence[str]) -> str:
    equations = extract_equations("\n".join(steps))
    payload = [equation["tree"] for equation in equations if equation["valid"]]
    if not payload:
        payload = [
            re.sub(r"\b(?:the|a|an|so|then|therefore)\b", "", step.lower())
            for step in steps
        ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def verify_generated_rationale(
    text: str,
    gold_answer: str,
    forbidden_fingerprints: Iterable[str],
) -> Tuple[Optional[dict], str]:
    predicted_answer = extract_answer(text)
    if predicted_answer is None:
        return None, "missing_final_answer"
    if not answers_match(predicted_answer, gold_answer):
        return None, "wrong_final_answer"
    steps = split_steps(text)
    if not steps:
        return None, "missing_steps"
    equations = extract_equations("\n".join(steps))
    if not equations:
        return None, "missing_executable_equation"
    if len(equations) != "\n".join(steps).count("="):
        return None, "unparsed_intermediate_equation"
    if any(not equation["valid"] for equation in equations):
        return None, "invalid_intermediate_equation"
    fingerprint = rationale_fingerprint(steps)
    if fingerprint in set(forbidden_fingerprints):
        return None, "surface_duplicate"
    return (
        {
            "steps": steps,
            "dependency_matrix": None,
            "confidence_matrix": None,
            "fingerprint": fingerprint,
            "source": f"qwen3_verified_{PROMPT_VERSION}",
            "verified": True,
            "verification": {
                "final_answer": predicted_answer,
                "n_executable_equations": len(equations),
                "all_equations_valid": True,
            },
        },
        "accepted",
    )


STRATEGIES = (
    "Combine dependent calculations into one nested arithmetic expression, "
    "without reproducing the reference equation sequence.",
    "Reorder independent sub-calculations and group terms differently from "
    "the reference route.",
    "Use a compact algebraic, ratio, or unit-rate formulation instead of the "
    "reference decomposition.",
    "Reason backward from the requested quantity or verify it by an inverse "
    "calculation when that is mathematically appropriate.",
    "Use a complement or total-minus-known decomposition rather than summing "
    "the same intermediate quantities in the reference order.",
    "Find another valid operation tree. If no genuinely different route "
    "exists, still solve correctly rather than inventing a false distinction.",
)


def build_prompt(item: dict, strategy: str) -> str:
    reference = "\n".join(
        f"{index + 1}. {step}" for index, step in enumerate(item["steps"])
    )
    return f"""Solve the math word problem using a concise, correct route that is structurally different from the reference route.

Problem:
{item["question"]}

Known correct final answer: {item["answer"]}

Reference route (do not merely paraphrase or copy its equation sequence):
{reference}

Requirements:
1. Use numbered reasoning steps.
2. Show an executable arithmetic equation in every calculation step.
3. Required strategy for this candidate: {strategy}
4. End with exactly `Answer: {item["answer"]}`.
5. Do not discuss these instructions or the reference route."""


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def completed_source_ids(path: Path, *, expected_split: str) -> set:
    if not path.is_file():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("prompt_version") != PROMPT_VERSION:
                    raise ValueError(
                        f"{path} contains a stale prompt version"
                    )
                if row.get("split") != expected_split:
                    raise ValueError(
                        f"{path} contains split={row.get('split')!r}, "
                        f"expected {expected_split!r}"
                    )
                completed.add(int(row["source_id"]))
    return completed


def generate_shard(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = load_json(args.source_dir / f"{args.split}.json")
    rows = [
        item
        for index, item in enumerate(source)
        if int(item.get("source_id", index)) % args.num_shards == args.shard_id
    ]
    if args.max_questions is not None:
        rows = rows[: int(args.max_questions)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_source_ids(
        args.output,
        expected_split=args.split,
    )
    rows = [
        item
        for index, item in enumerate(rows)
        if int(item.get("source_id", index)) not in completed
    ]
    if not rows:
        print(f"shard {args.shard_id}: already complete", flush=True)
        return

    torch.manual_seed(args.seed + args.shard_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as output_handle:
        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start : batch_start + args.batch_size]
            prompts = [
                build_prompt(
                    item,
                    STRATEGIES[candidate_index % len(STRATEGIES)],
                )
                for item in batch
                for candidate_index in range(args.candidates)
            ]
            rendered = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                try:
                    rendered.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                except TypeError:
                    rendered.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=1,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completions = tokenizer.batch_decode(
                generated[:, encoded.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            for local_index, item in enumerate(batch):
                start = local_index * args.candidates
                candidates = completions[start : start + args.candidates]
                gold_fingerprint = rationale_fingerprint(item["steps"])
                seen = {gold_fingerprint}
                accepted = []
                rejection_reasons = Counter()
                for candidate in candidates:
                    rationale, reason = verify_generated_rationale(
                        candidate,
                        str(item["answer"]),
                        seen,
                    )
                    if rationale is None:
                        rejection_reasons[reason] += 1
                        continue
                    accepted.append(rationale)
                    seen.add(rationale["fingerprint"])
                    if len(accepted) >= args.max_alternatives:
                        break
                record = {
                    "source_id": int(item.get("source_id", batch_start + local_index)),
                    "split": args.split,
                    "prompt_version": PROMPT_VERSION,
                    "gold_fingerprint": gold_fingerprint,
                    "accepted": accepted,
                    "rejection_reasons": dict(rejection_reasons),
                    "n_generated": len(candidates),
                }
                output_handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                output_handle.flush()
            print(
                f"shard {args.shard_id}: "
                f"{min(batch_start + len(batch), len(rows))}/{len(rows)}",
                flush=True,
            )


def load_shard_rows(
    paths: Sequence[Path],
    *,
    expected_split: Optional[str] = None,
) -> dict:
    merged = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("prompt_version") != PROMPT_VERSION:
                    raise ValueError(
                        f"{path} contains a stale prompt version"
                    )
                if (
                    expected_split is not None
                    and row.get("split") != expected_split
                ):
                    raise ValueError(
                        f"{path} contains split={row.get('split')!r}, "
                        f"expected {expected_split!r}"
                    )
                source_id = int(row["source_id"])
                if source_id in merged:
                    raise ValueError(f"Duplicate source_id {source_id}")
                merged[source_id] = row
    return merged


def merge_dataset(args) -> None:
    shard_paths = [
        args.shard_dir / f"rationales_shard_{index}.jsonl"
        for index in range(args.num_shards)
    ]
    generated = load_shard_rows(shard_paths)
    train = load_json(args.source_dir / "train.json")
    if len(generated) != len(train):
        raise ValueError(
            f"Expected {len(train)} generated rows, found {len(generated)}"
        )

    merged_train = []
    histogram = Counter()
    rejection_reasons = Counter()
    for index, item in enumerate(train):
        source_id = int(item.get("source_id", index))
        row = generated[source_id]
        alternatives = row.get("accepted", [])[: args.max_alternatives]
        updated = dict(item)
        updated["rationale_fingerprint"] = rationale_fingerprint(item["steps"])
        updated["rationale_source"] = "gold"
        updated["rationale_set"] = alternatives
        merged_train.append(updated)
        histogram[1 + len(alternatives)] += 1
        rejection_reasons.update(row.get("rejection_reasons", {}))

    multi_count = sum(
        count for n_rationales, count in histogram.items() if n_rationales >= 2
    )
    multi_fraction = multi_count / max(1, len(merged_train))
    if multi_fraction < args.min_multi_fraction:
        raise ValueError(
            f"Only {multi_fraction:.3%} of questions have a verified "
            f"alternative; required {args.min_multi_fraction:.3%}"
        )

    write_json(args.output_dir / "train.json", merged_train)
    for split in ("val", "test"):
        rows = load_json(args.source_dir / f"{split}.json")
        prepared = []
        for item in rows:
            updated = dict(item)
            updated["rationale_fingerprint"] = rationale_fingerprint(
                item["steps"]
            )
            updated["rationale_source"] = "gold"
            updated["rationale_set"] = []
            prepared.append(updated)
        write_json(args.output_dir / f"{split}.json", prepared)

    audit = {
        "status": "PASS",
        "prompt_version": PROMPT_VERSION,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "split_counts": {
            split: len(load_json(args.output_dir / f"{split}.json"))
            for split in ("train", "val", "test")
        },
        "rationale_count_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "questions_with_multiple_verified_rationales": multi_count,
        "multi_rationale_fraction": multi_fraction,
        "rejection_reasons": dict(rejection_reasons),
        "invariants": {
            "gold_rationale_retained": True,
            "generated_rationales_answer_verified": True,
            "generated_intermediate_equations_executed": True,
            "operation_fingerprints_deduplicated": True,
            "unverified_candidates_excluded": True,
        },
    }
    write_json(args.output_dir / "rationale_set_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def _merge_generated_split(
    *,
    source_rows: Sequence[dict],
    generated: dict,
    max_alternatives: int,
) -> Tuple[List[dict], Counter, Counter]:
    if len(generated) != len(source_rows):
        raise ValueError(
            f"Expected {len(source_rows)} generated rows, found {len(generated)}"
        )
    merged = []
    histogram = Counter()
    rejection_reasons = Counter()
    for index, item in enumerate(source_rows):
        source_id = int(item.get("source_id", index))
        if source_id not in generated:
            raise ValueError(f"Missing generated source_id {source_id}")
        row = generated[source_id]
        alternatives = row.get("accepted", [])[: int(max_alternatives)]
        updated = dict(item)
        updated["rationale_fingerprint"] = rationale_fingerprint(item["steps"])
        updated["rationale_source"] = "gold"
        updated["rationale_set"] = alternatives
        merged.append(updated)
        histogram[1 + len(alternatives)] += 1
        rejection_reasons.update(row.get("rejection_reasons", {}))
    return merged, histogram, rejection_reasons


def merge_all_dataset(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_audits = {}
    all_rejection_reasons = Counter()
    for split in ("train", "val", "test"):
        shard_dir = args.shard_root / split
        shard_paths = [
            shard_dir / f"rationales_shard_{index}.jsonl"
            for index in range(args.num_shards)
        ]
        generated = load_shard_rows(
            shard_paths,
            expected_split=split,
        )
        source_rows = load_json(args.source_dir / f"{split}.json")
        merged, histogram, rejection_reasons = _merge_generated_split(
            source_rows=source_rows,
            generated=generated,
            max_alternatives=args.max_alternatives,
        )
        multi_count = sum(
            count
            for n_rationales, count in histogram.items()
            if n_rationales >= 2
        )
        multi_fraction = multi_count / max(1, len(merged))
        minimum = (
            args.min_train_multi_fraction
            if split == "train"
            else args.min_eval_multi_fraction
        )
        if multi_fraction < minimum:
            raise ValueError(
                f"{split}: only {multi_fraction:.3%} of questions have a "
                f"verified alternative; required {minimum:.3%}"
            )
        write_json(args.output_dir / f"{split}.json", merged)
        all_rejection_reasons.update(rejection_reasons)
        split_audits[split] = {
            "count": len(merged),
            "rationale_count_histogram": {
                str(key): histogram[key] for key in sorted(histogram)
            },
            "questions_with_multiple_verified_rationales": multi_count,
            "multi_rationale_fraction": multi_fraction,
            "minimum_multi_rationale_fraction": minimum,
        }

    audit = {
        "status": "PASS",
        "prompt_version": PROMPT_VERSION,
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "generated_splits": ["train", "val", "test"],
        "splits": split_audits,
        "split_counts": {
            split: split_audits[split]["count"]
            for split in ("train", "val", "test")
        },
        "rejection_reasons": dict(all_rejection_reasons),
        "invariants": {
            "gold_rationale_retained": True,
            "generated_rationales_answer_verified": True,
            "generated_intermediate_equations_executed": True,
            "operation_fingerprints_deduplicated": True,
            "unverified_candidates_excluded": True,
            "heldout_rationales_analysis_only": True,
        },
    }
    write_json(args.output_dir / "rationale_set_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def bootstrap_gold_only(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = load_json(args.source_dir / f"{split}.json")
        prepared = []
        for item in rows:
            updated = dict(item)
            updated["rationale_fingerprint"] = rationale_fingerprint(
                item["steps"]
            )
            updated["rationale_source"] = "gold"
            updated["rationale_set"] = []
            prepared.append(updated)
        write_json(args.output_dir / f"{split}.json", prepared)
    print(f"Wrote gold-only smoke dataset to {args.output_dir}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-shard")
    generate.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    generate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
    )
    generate.add_argument("--shard-id", type=int, required=True)
    generate.add_argument("--num-shards", type=int, default=4)
    generate.add_argument("--batch-size", type=int, default=4)
    generate.add_argument("--max-questions", type=int)
    generate.add_argument("--candidates", type=int, default=3)
    generate.add_argument("--max-alternatives", type=int, default=3)
    generate.add_argument("--max-prompt-tokens", type=int, default=512)
    generate.add_argument("--max-new-tokens", type=int, default=192)
    generate.add_argument("--temperature", type=float, default=0.85)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=20260718)
    generate.set_defaults(handler=generate_shard)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    merge.add_argument("--shard-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--num-shards", type=int, default=4)
    merge.add_argument("--max-alternatives", type=int, default=3)
    merge.add_argument("--min-multi-fraction", type=float, default=0.20)
    merge.set_defaults(handler=merge_dataset)

    merge_all = subparsers.add_parser("merge-all")
    merge_all.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    merge_all.add_argument("--shard-root", type=Path, required=True)
    merge_all.add_argument("--output-dir", type=Path, required=True)
    merge_all.add_argument("--num-shards", type=int, default=4)
    merge_all.add_argument("--max-alternatives", type=int, default=3)
    merge_all.add_argument(
        "--min-train-multi-fraction",
        type=float,
        default=0.20,
    )
    merge_all.add_argument(
        "--min-eval-multi-fraction",
        type=float,
        default=0.20,
    )
    merge_all.set_defaults(handler=merge_all_dataset)

    bootstrap = subparsers.add_parser("bootstrap-gold-only")
    bootstrap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    bootstrap.add_argument("--output-dir", type=Path, required=True)
    bootstrap.set_defaults(handler=bootstrap_gold_only)
    return root


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "shard_id") and not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must be in [0, num-shards)")
    args.handler(args)


if __name__ == "__main__":
    main()
