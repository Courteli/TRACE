#!/usr/bin/env python3
"""Build the offline TRACE-VB gold-prefix sufficiency cache.

The cache is intentionally produced outside the training process.  For every
registered GSM8K example it scores the gold answer after the question alone
and after every *gold* CoT prefix under a frozen Stage-0 teacher.  Scores are
normalised between the question-only and full-CoT endpoints.  Prefixes at and
after the first literal gold-answer occurrence are masked, rather than being
used as process supervision.

The module keeps model imports inside ``load_teacher`` so its data, masking,
normalisation, fingerprint, and audit functions can be unit-tested without
initialising or downloading a language model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "trace_vb_prefix_sufficiency_v1"
CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(
    "/disk1/dingxukai/TRACE/data/raw/GSM8k-Aug-NL/"
    "gsm8k_train_processed.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/disk1/dingxukai/TRACE/trace_vb_runs/cache/"
    "gsm8k_prefix_sufficiency_v1.pt"
)
REGISTERED_GSM8K_FILES = {
    "gsm8k_train_processed.jsonl": (
        6726,
        "31e256348cb35ef34bb63c66339a2b8483be44896547c2644d31c0d92e56540c",
    ),
    "gsm8k_val_processed.jsonl": (
        747,
        "c9ef2ef23b44ea661577e5eb02456738b02a133d70a8b29e57adf86342da0b4f",
    ),
    "gsm8k_test_processed.jsonl": (
        1319,
        "5395be51d54d7af531883af873e130d3f280a7e5d9aaf849e7c20ed356e3847b",
    ),
}
REGISTERED_GSM8K_DIR = Path(
    "/disk1/dingxukai/TRACE/data/raw/GSM8k-Aug-NL"
)

QUESTION_TEMPLATE = "Question: {} Let's think step by step:"
SPEED_SUFFIX = "(Thinking speed: 1)"
THINKING_SEPARATOR = "###"
ANSWER_TEMPLATE = "Answer:{}"
PROMPT_SPEC = {
    "question_template": QUESTION_TEMPLATE,
    "speed_suffix": SPEED_SUFFIX,
    "thinking_separator": THINKING_SEPARATOR,
    "answer_template": ANSWER_TEMPLATE,
    "question_only_context": "question + speed + separator",
    "prefix_context": "question + speed + separator + prefix + separator",
    "prefix_join": "newline between deterministic CoT sentence steps",
    "target": "Answer:{gold_answer} + eos",
    "tokenisation": (
        "question, reasoning chunk, and answer target encoded separately, "
        "then concatenated exactly as Stage-0 LitCot.forward"
    ),
    "logp": "mean causal log-probability over target tokens including eos",
}
TOKENIZER_RUNTIME_SPEC = {
    "trust_remote_code": False,
    "local_files_only": True,
    "pad_token_added_by_stage0": "[PAD]",
    "add_special_tokens": False,
}
PRE_ACTION_ROLE_NAMES = (
    "PLAN",
    "SOLVE1",
    "SOLVE2",
    "SOLVE3",
    "SOLVE4",
    "SOLVE5",
    "REFINE",
    "COMMIT",
)

TOKENIZER_FILE_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint tokenizer assets without hashing base-model weights."""

    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Base-model/tokenizer directory not found: {path}")
    files = []
    for candidate in sorted(path.iterdir(), key=lambda item: item.name):
        if not candidate.is_file():
            continue
        if (
            candidate.name in TOKENIZER_FILE_NAMES
            or candidate.name.startswith("tokenization_")
        ):
            files.append(
                {
                    "name": candidate.name,
                    "size": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not files:
        raise ValueError(f"No tokenizer assets found in {path}")
    manifest = {
        "path": str(path),
        "files": files,
        "runtime_spec": TOKENIZER_RUNTIME_SPEC,
    }
    fingerprint_payload = {
        "files": files,
        "runtime_spec": TOKENIZER_RUNTIME_SPEC,
    }
    return {"sha256": sha256_json(fingerprint_payload), "manifest": manifest}


def make_fingerprints(
    *,
    data_path: Path,
    teacher_checkpoint: Path,
    base_model: Path,
    prompt_spec: Mapping[str, Any] = PROMPT_SPEC,
) -> dict[str, Any]:
    """Create all fingerprints required for fail-fast cache loading."""

    data_path = Path(data_path).resolve()
    checkpoint = Path(teacher_checkpoint).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    tokenizer = tokenizer_fingerprint(base_model)
    components = {
        "data_sha256": sha256_file(data_path),
        "teacher_checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_sha256": tokenizer["sha256"],
        "prompt_sha256": sha256_json(prompt_spec),
    }
    return {
        **components,
        "composite_sha256": sha256_json(components),
        "tokenizer_manifest": tokenizer["manifest"],
    }


_TITLE_ABBREVIATIONS = {
    "dr",
    "jr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "sr",
    "st",
}
_UNIT_ABBREVIATIONS = {
    "cm",
    "ft",
    "hr",
    "hrs",
    "in",
    "kg",
    "km",
    "lb",
    "lbs",
    "min",
    "mins",
    "oz",
    "sec",
    "secs",
    "sq",
}


def _is_cot_sentence_boundary(text: str, punctuation_index: int) -> bool:
    if text[punctuation_index] in "!?":
        return True
    prefix = text[: punctuation_index + 1]
    suffix = text[punctuation_index + 1 :].lstrip()
    if not suffix:
        return True
    if re.search(r"(?:\b[A-Za-z]\.){2,}$", prefix):
        return False
    token_match = re.search(r"([A-Za-z]+)\.$", prefix)
    if token_match is None:
        return True
    token = token_match.group(1)
    lowered = token.lower()
    if lowered in _TITLE_ABBREVIATIONS or len(token) == 1:
        return False
    if lowered in _UNIT_ABBREVIATIONS:
        return suffix[0].isupper()
    return True


def split_cot_steps(cot: str) -> list[str]:
    """Mirror the registered GSM8K dataset's deterministic sentence split."""

    raw = str(cot).replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [
        re.sub(r"[ \t]+", " ", part).strip()
        for part in re.split(r"\n+", raw)
        if part.strip()
    ]
    steps = []
    for paragraph in paragraphs:
        start = 0
        for match in re.finditer(r"[.!?](?=\s+|$)", paragraph):
            punctuation_index = match.start()
            if not _is_cot_sentence_boundary(paragraph, punctuation_index):
                continue
            part = paragraph[start : punctuation_index + 1].strip()
            if part:
                steps.append(part)
            start = punctuation_index + 1
        tail = paragraph[start:].strip()
        if tail:
            steps.append(tail)
    return steps or [raw or "\n"]


def cumulative_prefixes(steps: Sequence[str]) -> list[str]:
    return [
        "\n".join(str(part).strip() for part in steps[: index + 1])
        for index in range(len(steps))
    ]


def solve_slot_indices(
    n_cot_steps: int, *, n_solve_slots: int = 5
) -> tuple[int, ...]:
    """Mirror TRACE's deterministic short-CoT placement over SOLVE slots."""

    if int(n_cot_steps) <= 0 or int(n_solve_slots) <= 0:
        raise ValueError("step and slot counts must be positive")
    active = min(int(n_cot_steps), int(n_solve_slots))
    if active == int(n_solve_slots):
        return tuple(range(int(n_solve_slots)))
    if active == 1:
        return (int(n_solve_slots) - 1,)
    denominator = active - 1
    final_slot = int(n_solve_slots) - 1
    return tuple(
        (index * final_slot + denominator // 2) // denominator
        for index in range(active)
    )


def contiguous_cot_chunk_spans(
    n_cot_steps: int, *, n_chunks: int = 5
) -> tuple[tuple[int, int], ...]:
    """Return the exact monotone CoT-to-SOLVE partition used by TRACE."""

    if int(n_cot_steps) <= 0 or int(n_chunks) <= 0:
        raise ValueError("step and chunk counts must be positive")
    slots = solve_slot_indices(n_cot_steps, n_solve_slots=n_chunks)
    spans = []
    start = 0
    if int(n_cot_steps) < int(n_chunks):
        active_slots = set(slots)
        for chunk_index in range(int(n_chunks)):
            width = int(chunk_index in active_slots)
            end = start + width
            spans.append((start, end))
            start = end
    else:
        base, remainder = divmod(int(n_cot_steps), int(n_chunks))
        for chunk_index in range(int(n_chunks)):
            width = base + int(chunk_index < remainder)
            end = start + width
            spans.append((start, end))
            start = end
    if start != int(n_cot_steps):
        raise RuntimeError("CoT partition did not cover every observed step")
    return tuple(spans)


def align_prefix_signal_to_pre_action_roles(
    scores: Sequence[float],
    valid_mask: Sequence[bool],
) -> dict[str, list[Any]]:
    """Align textual-prefix sufficiency with the eight critic input states.

    A critic value is attached to the state *before* the named action. PLAN's
    input is the question-only state.  PLAN adds no observed textual fact, so
    SOLVE1's input has the same zero endpoint.  SOLVE2..5 use the endpoint of
    the preceding SOLVE chunk, REFINE uses the SOLVE5 endpoint, and COMMIT uses
    the full-prefix endpoint associated with REFINE. Empty SOLVE chunks and
    leaked endpoints are masked; no value is copied across an unobserved slot.
    """

    if len(scores) != len(valid_mask) or not scores:
        raise ValueError("scores and valid_mask must be equal non-zero lengths")
    spans = contiguous_cot_chunk_spans(len(scores), n_chunks=5)
    slot_sources: list[int | None] = [
        end - 1 if end > start else None for start, end in spans
    ]
    denominator_valid = any(bool(value) for value in valid_mask)

    # Source -1 denotes the normalised question-only endpoint, exactly 0.
    sources: list[int | None] = [-1, -1]
    sources.extend(slot_sources[:4])
    sources.append(slot_sources[4])
    sources.append(len(scores) - 1)
    if len(sources) != len(PRE_ACTION_ROLE_NAMES):
        raise RuntimeError("pre-action role alignment is not eight positions")

    role_scores = []
    role_mask = []
    for source in sources:
        if source == -1:
            role_scores.append(0.0)
            role_mask.append(denominator_valid)
        elif source is None:
            role_scores.append(0.0)
            role_mask.append(False)
        else:
            role_scores.append(float(scores[source]))
            role_mask.append(bool(valid_mask[source]))
    return {
        "role_scores": role_scores,
        "role_valid_mask": role_mask,
        "role_source_prefix_index": sources,
    }


_NUMBER_TOKEN = re.compile(
    r"(?<![\d.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?!\d|\.\d)"
)


def _as_decimal(value: str) -> Decimal | None:
    cleaned = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("$", "")
        .replace("£", "")
        .replace("€", "")
        .replace("−", "-")
    )
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def contains_explicit_answer(text: str, answer: str) -> bool:
    """Detect a literal gold answer, with numeric boundary normalisation."""

    answer_decimal = _as_decimal(answer)
    if answer_decimal is not None:
        for match in _NUMBER_TOKEN.finditer(str(text).replace("−", "-")):
            candidate = _as_decimal(match.group(0))
            if candidate is not None and candidate == answer_decimal:
                return True
        return False
    answer_text = re.sub(r"\s+", " ", str(answer).strip()).casefold()
    if not answer_text:
        return False
    normalised_text = re.sub(r"\s+", " ", str(text)).casefold()
    pattern = rf"(?<!\w){re.escape(answer_text)}(?!\w)"
    return re.search(pattern, normalised_text) is not None


def first_answer_leakage_step(
    steps: Sequence[str], answer: str
) -> int | None:
    """Return the zero-based first prefix containing the literal answer."""

    for index, prefix in enumerate(cumulative_prefixes(steps)):
        if contains_explicit_answer(prefix, answer):
            return index
    return None


def build_question_context(question: str) -> str:
    return (
        QUESTION_TEMPLATE.format(str(question))
        + SPEED_SUFFIX
        + THINKING_SEPARATOR
    )


def build_prefix_context(question: str, prefix: str) -> str:
    return (
        QUESTION_TEMPLATE.format(str(question))
        + SPEED_SUFFIX
        + THINKING_SEPARATOR
        + str(prefix)
        + THINKING_SEPARATOR
    )


def build_answer_target(answer: str, eos_token: str) -> str:
    return ANSWER_TEMPLATE.format(str(answer)) + str(eos_token)


def encode_scoring_context(
    tokenizer: Any,
    question: str,
    prefix: str | None,
) -> list[int]:
    """Encode context with the same chunk boundaries used by Stage-0 SFT."""

    question_chunk = QUESTION_TEMPLATE.format(str(question)) + SPEED_SUFFIX
    reasoning_chunk = (
        THINKING_SEPARATOR
        if prefix is None
        else THINKING_SEPARATOR + str(prefix) + THINKING_SEPARATOR
    )
    return list(
        tokenizer.encode(question_chunk, add_special_tokens=False)
    ) + list(tokenizer.encode(reasoning_chunk, add_special_tokens=False))


@dataclass(frozen=True)
class NormalisedSignal:
    scores_raw: list[float | None]
    scores: list[float]
    valid_mask: list[bool]
    denominator: float | None
    invalid_reason: str | None


def normalise_prefix_logps(
    question_logp: float,
    prefix_logps: Sequence[float],
    *,
    leakage_step: int | None,
    min_full_improvement: float = 1e-4,
    max_full_improvement: float = 50.0,
    max_abs_raw_score: float = 8.0,
) -> NormalisedSignal:
    """Normalise prefix logps and construct a strict supervision mask.

    The full-prefix endpoint may contain the answer and is still used only as
    a normalisation anchor.  The leaking prefix itself and all later prefixes
    are invalid supervision targets.
    """

    count = len(prefix_logps)
    if count == 0:
        return NormalisedSignal([], [], [], None, "no_prefixes")
    values = [float(value) for value in prefix_logps]
    if not math.isfinite(float(question_logp)) or not all(
        math.isfinite(value) for value in values
    ):
        return NormalisedSignal(
            [None] * count,
            [0.0] * count,
            [False] * count,
            None,
            "nonfinite_logp",
        )
    denominator = values[-1] - float(question_logp)
    if denominator <= min_full_improvement:
        return NormalisedSignal(
            [None] * count,
            [0.0] * count,
            [False] * count,
            denominator,
            "full_not_better_than_question",
        )
    if denominator > max_full_improvement:
        return NormalisedSignal(
            [None] * count,
            [0.0] * count,
            [False] * count,
            denominator,
            "anomalous_denominator",
        )

    raw = [(value - float(question_logp)) / denominator for value in values]
    scores = [min(1.0, max(0.0, value)) for value in raw]
    valid = []
    for index, value in enumerate(raw):
        before_leakage = leakage_step is None or index < leakage_step
        valid.append(before_leakage and abs(value) <= max_abs_raw_score)
    raw_output: list[float | None] = [
        value if math.isfinite(value) else None for value in raw
    ]
    reason = None if any(valid) else "no_pre_leakage_prefix"
    return NormalisedSignal(raw_output, scores, valid, denominator, reason)


def load_registered_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Validate and load one immutable registered GSM8K split."""

    path = Path(path).resolve()
    if path.parent != REGISTERED_GSM8K_DIR.resolve():
        raise ValueError(
            "TRACE-VB sufficiency cache only accepts the immutable registered "
            f"GSM8K directory: {REGISTERED_GSM8K_DIR}"
        )
    if path.name not in REGISTERED_GSM8K_FILES:
        raise ValueError(f"Unregistered GSM8K file: {path.name}")
    expected_count, expected_hash = REGISTERED_GSM8K_FILES[path.name]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{path} SHA256 {actual_hash}, expected {expected_hash}"
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            missing = {"question", "cot", "answer"} - set(row)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} lacks fields {sorted(missing)}"
                )
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_count}")
    source_ids = [int(row.get("id", index)) for index, row in enumerate(rows)]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{path} contains duplicate source ids")
    return rows, actual_hash


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarise_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute fixed fail-fast statistics from serialisable cache rows."""

    total_rows = len(rows)
    valid_rows = 0
    prefixes_total = 0
    valid_prefixes = 0
    leakage_rows = 0
    spans = []
    nonzero_gain_rows = 0
    denominators = []
    reasons: Counter[str] = Counter()
    for row in rows:
        mask = [bool(value) for value in row.get("valid_mask", [])]
        scores = [float(value) for value in row.get("scores", [])]
        prefixes_total += len(mask)
        valid_indices = [index for index, keep in enumerate(mask) if keep]
        valid_prefixes += len(valid_indices)
        if valid_indices:
            valid_rows += 1
            selected = [scores[index] for index in valid_indices]
            span = max(selected) - min(selected) if len(selected) > 1 else 0.0
            spans.append(span)
            gains = [selected[0]] + [
                selected[index] - selected[index - 1]
                for index in range(1, len(selected))
            ]
            if any(abs(value) > 1e-4 for value in gains):
                nonzero_gain_rows += 1
        if row.get("leakage_step") is not None:
            leakage_rows += 1
        denominator = row.get("denominator")
        if denominator is not None and math.isfinite(float(denominator)):
            denominators.append(float(denominator))
        reason = row.get("invalid_reason")
        if reason:
            reasons[str(reason)] += 1
    return {
        "rows_total": total_rows,
        "valid_rows": valid_rows,
        "valid_row_fraction": valid_rows / total_rows if total_rows else 0.0,
        "prefixes_total": prefixes_total,
        "valid_prefixes": valid_prefixes,
        "valid_prefix_fraction": (
            valid_prefixes / prefixes_total if prefixes_total else 0.0
        ),
        "leakage_rows": leakage_rows,
        "leakage_rate": leakage_rows / total_rows if total_rows else 0.0,
        "pre_leakage_score_span_mean": (
            statistics.fmean(spans) if spans else 0.0
        ),
        "pre_leakage_score_span_median": (
            statistics.median(spans) if spans else 0.0
        ),
        "pre_leakage_score_span_p90": percentile(spans, 0.90),
        "pre_leakage_nonzero_gain_rows": nonzero_gain_rows,
        "pre_leakage_nonzero_gain_fraction": (
            nonzero_gain_rows / valid_rows if valid_rows else 0.0
        ),
        "denominator_mean": (
            statistics.fmean(denominators) if denominators else None
        ),
        "denominator_min": min(denominators) if denominators else None,
        "denominator_max": max(denominators) if denominators else None,
        "invalid_reason_counts": dict(sorted(reasons.items())),
    }


def audit_stats(
    stats: Mapping[str, Any],
    *,
    min_valid_row_fraction: float,
    min_valid_prefix_fraction: float,
    min_nonzero_gain_fraction: float,
    min_score_span_mean: float,
) -> list[str]:
    failures = []
    checks = (
        ("valid_row_fraction", min_valid_row_fraction),
        ("valid_prefix_fraction", min_valid_prefix_fraction),
        ("pre_leakage_nonzero_gain_fraction", min_nonzero_gain_fraction),
        ("pre_leakage_score_span_mean", min_score_span_mean),
    )
    for name, minimum in checks:
        actual = float(stats.get(name, 0.0))
        if actual < float(minimum):
            failures.append(f"{name}={actual:.6f} < {float(minimum):.6f}")
    return failures


def _dtype_from_name(torch_module: Any, name: str) -> Any:
    aliases = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    return aliases[name]


def _normalise_checkpoint_key(name: str) -> str:
    for prefix in ("_forward_module.", "module.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if name.startswith("llm."):
        name = name[len("llm.") :]
    return name


def load_teacher(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    """Load a local Qwen base plus the Lightning Stage-0 LoRA weights."""

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = Path(args.base_model).resolve()
    checkpoint_path = Path(args.teacher_checkpoint).resolve()
    if not base_model.is_dir():
        raise FileNotFoundError(base_model)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    dtype = _dtype_from_name(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        trust_remote_code=False,
        local_files_only=True,
    )
    # LitCoTModelBase unconditionally installs this pad token before loading
    # the model, even when the base tokenizer already declares another pad.
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        trust_remote_code=False,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    target_modules = [
        item.strip() for item in args.lora_target_modules.split(",") if item.strip()
    ]
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("Teacher checkpoint has no state_dict mapping")
    model_state = model.state_dict()
    selected = {}
    for original_name, tensor in raw_state.items():
        name = _normalise_checkpoint_key(str(original_name))
        if name in model_state and tuple(model_state[name].shape) == tuple(tensor.shape):
            selected[name] = tensor
    lora_selected = [name for name in selected if "lora_" in name]
    lora_expected = [name for name in model_state if "lora_" in name]
    if not lora_selected:
        sample = list(raw_state)[:5]
        raise ValueError(
            "No Stage-0 LoRA tensors matched the base model. "
            f"Checkpoint key sample: {sample}"
        )
    missing_lora = sorted(set(lora_expected) - set(lora_selected))
    if missing_lora:
        raise ValueError(
            f"Teacher checkpoint is missing {len(missing_lora)} LoRA tensors; "
            f"first missing key: {missing_lora[0]}"
        )
    model.load_state_dict(selected, strict=False)
    del checkpoint, raw_state, selected, model_state
    model.requires_grad_(False)
    model.eval()
    model.to(args.device)
    load_info = {
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_target_modules": target_modules,
        "lora_tensor_count": len(lora_selected),
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
    }
    return model, tokenizer, load_info


def score_contexts(
    *,
    model: Any,
    tokenizer: Any,
    context_token_ids: Sequence[Sequence[int]],
    answer: str,
    device: str,
    batch_size: int,
) -> list[float]:
    """Return mean gold-target logp for independently encoded contexts."""

    import torch
    import torch.nn.functional as torch_functional

    eos = tokenizer.eos_token
    if eos is None:
        raise ValueError("Teacher tokenizer has no eos_token")
    target_ids = tokenizer.encode(
        build_answer_target(answer, eos),
        add_special_tokens=False,
    )
    if not target_ids:
        raise ValueError("Gold answer target tokenised to zero tokens")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Teacher tokenizer has no pad_token_id")
    results = []
    for start in range(0, len(context_token_ids), batch_size):
        context_ids = [
            [int(token_id) for token_id in ids]
            for ids in context_token_ids[start : start + batch_size]
        ]
        sequences = [ids + target_ids for ids in context_ids]
        maximum = max(len(ids) for ids in sequences)
        input_ids = torch.full(
            (len(sequences), maximum),
            int(pad_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_index, (sequence, context) in enumerate(
            zip(sequences, context_ids)
        ):
            length = len(sequence)
            input_ids[row_index, :length] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )
            attention_mask[row_index, :length] = 1
            target_mask[row_index, len(context) : length] = True
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits
        shifted_logits = logits[:, :-1, :].float()
        shifted_targets = input_ids[:, 1:]
        shifted_mask = target_mask[:, 1:]
        token_logps = torch_functional.log_softmax(
            shifted_logits, dim=-1
        ).gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
        sums = (token_logps * shifted_mask).sum(dim=-1)
        counts = shifted_mask.sum(dim=-1)
        if torch.any(counts != len(target_ids)):
            raise RuntimeError("Answer target mask has an unexpected length")
        results.extend((sums / counts).detach().cpu().tolist())
    return [float(value) for value in results]


def make_cache_row(
    *,
    dataset_idx: int,
    source_row: Mapping[str, Any],
    logps: Sequence[float],
    min_full_improvement: float,
    max_full_improvement: float,
    max_abs_raw_score: float,
) -> dict[str, Any]:
    steps = split_cot_steps(str(source_row["cot"]))
    if len(logps) != len(steps) + 1:
        raise ValueError(
            f"Expected {len(steps) + 1} logps, received {len(logps)}"
        )
    leakage_step = first_answer_leakage_step(steps, str(source_row["answer"]))
    signal = normalise_prefix_logps(
        float(logps[0]),
        logps[1:],
        leakage_step=leakage_step,
        min_full_improvement=min_full_improvement,
        max_full_improvement=max_full_improvement,
        max_abs_raw_score=max_abs_raw_score,
    )
    role_signal = align_prefix_signal_to_pre_action_roles(
        signal.scores, signal.valid_mask
    )
    return {
        "idx": int(dataset_idx),
        "source_id": int(source_row.get("id", dataset_idx)),
        "question_sha256": sha256_text(str(source_row["question"])),
        "n_steps": len(steps),
        "question_logp": float(logps[0]),
        "full_logp": float(logps[-1]),
        "prefix_logps": [float(value) for value in logps[1:]],
        "scores_raw": signal.scores_raw,
        "scores": signal.scores,
        "valid_mask": signal.valid_mask,
        "leakage_step": leakage_step,
        "leakage_prefix_number": (
            leakage_step + 1 if leakage_step is not None else None
        ),
        "denominator": signal.denominator,
        "invalid_reason": signal.invalid_reason,
        **role_signal,
    }


def validate_cache_object(cache: Mapping[str, Any]) -> None:
    if cache.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected cache schema: {cache.get('schema_version')!r}"
        )
    rows = cache.get("rows")
    by_idx = cache.get("by_idx")
    metadata = cache.get("metadata")
    if not isinstance(rows, list) or not isinstance(by_idx, Mapping):
        raise TypeError("Cache must contain rows:list and by_idx:mapping")
    if not isinstance(metadata, Mapping):
        raise TypeError("Cache must contain metadata")
    expected = {int(row["idx"]) for row in rows}
    actual = {int(index) for index in by_idx}
    if expected != actual or len(expected) != len(rows):
        raise ValueError("Cache by_idx does not exactly index unique rows")
    for row in rows:
        length = int(row["n_steps"])
        for name in ("prefix_logps", "scores_raw", "scores", "valid_mask"):
            if len(row[name]) != length:
                raise ValueError(f"row {row['idx']} has malformed {name}")
        for name in (
            "role_scores",
            "role_valid_mask",
            "role_source_prefix_index",
        ):
            if len(row[name]) != len(PRE_ACTION_ROLE_NAMES):
                raise ValueError(f"row {row['idx']} has malformed {name}")


def write_cache(cache: Mapping[str, Any], output: Path) -> None:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix not in {".pt", ".jsonl"}:
        raise ValueError("Cache output must end in .pt or .jsonl")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if output.suffix == ".pt":
            import torch

            torch.save(dict(cache), temporary)
        else:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(
                    canonical_json(
                        {
                            "_type": "metadata",
                            "schema_version": cache["schema_version"],
                            "metadata": cache["metadata"],
                        }
                    )
                    + "\n"
                )
                for row in cache["rows"]:
                    handle.write(
                        canonical_json({"_type": "row", **row}) + "\n"
                    )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_cache(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.suffix == ".pt":
        import torch

        cache = torch.load(path, map_location="cpu", weights_only=False)
    elif path.suffix == ".jsonl":
        metadata = None
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.pop("_type", None) == "metadata":
                    metadata = record
                else:
                    record.pop("_type", None)
                    rows.append(record)
        if metadata is None:
            raise ValueError("JSONL cache is missing its metadata record")
        cache = {
            "schema_version": metadata["schema_version"],
            "metadata": metadata["metadata"],
            "rows": rows,
            "by_idx": {int(row["idx"]): row for row in rows},
        }
    else:
        raise ValueError("Cache path must end in .pt or .jsonl")
    validate_cache_object(cache)
    return cache


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--teacher_checkpoint", type=Path)
    parser.add_argument("--base_model", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--min_full_improvement", type=float, default=1e-4)
    parser.add_argument("--max_full_improvement", type=float, default=50.0)
    parser.add_argument("--max_abs_raw_score", type=float, default=8.0)
    parser.add_argument("--min_valid_row_fraction", type=float, default=0.50)
    parser.add_argument("--min_valid_prefix_fraction", type=float, default=0.20)
    parser.add_argument("--min_nonzero_gain_fraction", type=float, default=0.30)
    parser.add_argument("--min_score_span_mean", type=float, default=0.05)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs/fingerprints and print the plan; do not load a model or write cache.",
    )
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="Read --output, recompute signal statistics, and apply audit thresholds.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.dry_run and args.audit_only:
        parser.error("--dry_run and --audit_only are mutually exclusive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if not args.audit_only:
        if args.teacher_checkpoint is None:
            parser.error("--teacher_checkpoint is required")
        if args.base_model is None:
            parser.error("--base_model is required")


def _audit_or_raise(args: argparse.Namespace, stats: Mapping[str, Any]) -> None:
    failures = audit_stats(
        stats,
        min_valid_row_fraction=args.min_valid_row_fraction,
        min_valid_prefix_fraction=args.min_valid_prefix_fraction,
        min_nonzero_gain_fraction=args.min_nonzero_gain_fraction,
        min_score_span_mean=args.min_score_span_mean,
    )
    if failures:
        raise RuntimeError("Sufficiency signal audit failed: " + "; ".join(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    if args.audit_only:
        cache = read_cache(args.output)
        stats = summarise_rows(cache["rows"])
        _print_json(
            {
                "status": "PASS" if not audit_stats(
                    stats,
                    min_valid_row_fraction=args.min_valid_row_fraction,
                    min_valid_prefix_fraction=args.min_valid_prefix_fraction,
                    min_nonzero_gain_fraction=args.min_nonzero_gain_fraction,
                    min_score_span_mean=args.min_score_span_mean,
                ) else "FAIL",
                "schema_version": cache["schema_version"],
                "output": str(Path(args.output).resolve()),
                "stats": stats,
            }
        )
        _audit_or_raise(args, stats)
        return 0

    source_rows, data_hash = load_registered_rows(args.data)
    fingerprints = make_fingerprints(
        data_path=args.data,
        teacher_checkpoint=args.teacher_checkpoint,
        base_model=args.base_model,
    )
    if fingerprints["data_sha256"] != data_hash:
        raise RuntimeError("Data changed while fingerprints were computed")
    selected_rows = source_rows[: args.limit] if args.limit else source_rows
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run" if args.dry_run else "build",
        "data": str(Path(args.data).resolve()),
        "registered_rows": len(source_rows),
        "selected_rows": len(selected_rows),
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "base_model": str(Path(args.base_model).resolve()),
        "output": str(Path(args.output).resolve()),
        "fingerprints": fingerprints,
        "prompt_spec": PROMPT_SPEC,
    }
    if args.dry_run:
        _print_json(plan)
        return 0

    random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, teacher_load = load_teacher(args)
    cache_rows = []
    for dataset_idx, source_row in enumerate(selected_rows):
        steps = split_cot_steps(str(source_row["cot"]))
        prefixes = cumulative_prefixes(steps)
        context_token_ids = [
            encode_scoring_context(
                tokenizer, str(source_row["question"]), prefix=None
            )
        ] + [
            encode_scoring_context(
                tokenizer, str(source_row["question"]), prefix=prefix
            )
            for prefix in prefixes
        ]
        logps = score_contexts(
            model=model,
            tokenizer=tokenizer,
            context_token_ids=context_token_ids,
            answer=str(source_row["answer"]),
            device=args.device,
            batch_size=args.batch_size,
        )
        cache_rows.append(
            make_cache_row(
                dataset_idx=dataset_idx,
                source_row=source_row,
                logps=logps,
                min_full_improvement=args.min_full_improvement,
                max_full_improvement=args.max_full_improvement,
                max_abs_raw_score=args.max_abs_raw_score,
            )
        )
        if (dataset_idx + 1) % 100 == 0 or dataset_idx + 1 == len(selected_rows):
            print(
                f"scored {dataset_idx + 1}/{len(selected_rows)} rows",
                file=sys.stderr,
                flush=True,
            )
    stats = summarise_rows(cache_rows)
    cache = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "builder": str(Path(__file__).resolve()),
            "fingerprints": fingerprints,
            "paths": {
                "data": str(Path(args.data).resolve()),
                "teacher_checkpoint": str(
                    Path(args.teacher_checkpoint).resolve()
                ),
                "base_model": str(Path(args.base_model).resolve()),
            },
            "prompt_spec": PROMPT_SPEC,
            "pre_action_role_alignment": {
                "role_names": list(PRE_ACTION_ROLE_NAMES),
                "rule": (
                    "PLAN=q_only; SOLVE1=post_PLAN(q_only); "
                    "SOLVE2..5=preceding_SOLVE_chunk_endpoint; "
                    "REFINE=SOLVE5_endpoint; COMMIT=full_prefix_endpoint"
                ),
                "question_only_source_index": -1,
                "empty_chunk_source_index": None,
                "leaked_or_empty_endpoints_are_masked": True,
            },
            "teacher_load": teacher_load,
            "normalisation": {
                "formula": "(prefix_logp-question_logp)/(full_logp-question_logp)",
                "clip": [0.0, 1.0],
                "min_full_improvement": args.min_full_improvement,
                "max_full_improvement": args.max_full_improvement,
                "max_abs_raw_score": args.max_abs_raw_score,
                "mask_from_first_literal_answer_prefix": True,
            },
            "selection": {
                "registered_rows": len(source_rows),
                "cached_rows": len(cache_rows),
                "limit": args.limit,
                "seed": args.seed,
            },
            "stats": stats,
        },
        "rows": cache_rows,
        "by_idx": {int(row["idx"]): row for row in cache_rows},
    }
    validate_cache_object(cache)
    write_cache(cache, args.output)
    _print_json(
        {
            "status": "BUILT",
            "output": str(Path(args.output).resolve()),
            "schema_version": SCHEMA_VERSION,
            "stats": stats,
            "audit_failures": audit_stats(
                stats,
                min_valid_row_fraction=args.min_valid_row_fraction,
                min_valid_prefix_fraction=args.min_valid_prefix_fraction,
                min_nonzero_gain_fraction=args.min_nonzero_gain_fraction,
                min_score_span_mean=args.min_score_span_mean,
            ),
        }
    )
    _audit_or_raise(args, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
