"""Ordinary CoT JSONL only. No graph, dependency or confidence annotations."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


SPLITS = {
    "train": ("gsm8k_train_processed.jsonl", 6726, "31e256348cb35ef34bb63c66339a2b8483be44896547c2644d31c0d92e56540c"),
    "val": ("gsm8k_val_processed.jsonl", 747, "c9ef2ef23b44ea661577e5eb02456738b02a133d70a8b29e57adf86342da0b4f"),
    "test": ("gsm8k_test_processed.jsonl", 1319, "5395be51d54d7af531883af873e130d3f280a7e5d9aaf849e7c20ed356e3847b"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    steps: tuple[str, ...]
    answer: str


def parse_row(row, split):
    if set(row) != {"id", "question", "cot", "answer"}:
        raise ValueError("expected exactly id/question/cot/answer; annotated QSA/graph rows are not accepted")
    if not isinstance(row["id"], (str, int)) or isinstance(row["id"], bool):
        raise ValueError("invalid source ID")
    if any(not isinstance(row[k], str) or not row[k].strip() for k in ("question", "cot", "answer")):
        raise ValueError("question, cot and answer must be non-empty strings")
    steps = tuple(line.strip() for line in row["cot"].split("\n") if line.strip())
    if not steps:
        raise ValueError("empty CoT is not a training target")
    return Example(f"{split}:{row['id']}", row["question"].strip(), steps, row["answer"].strip())


def load_jsonl(path, split, expected_hash=None, expected_count=None):
    path = Path(path).resolve(strict=True)
    if path.suffix != ".jsonl" or any(p == "readcot_qsa_qwen_dc" for p in path.parts):
        raise ValueError("only ordinary CoT JSONL files are accepted")
    if expected_hash and file_hash(path) != expected_hash:
        raise ValueError(f"data fingerprint mismatch: {path.name}")
    # Unicode line separators can legitimately occur INSIDE JSON strings.
    # Iterate physical JSONL lines instead of str.splitlines().
    with path.open(encoding="utf-8") as stream:
        examples = [parse_row(json.loads(line), split) for line in stream if line.strip()]
    if expected_count is not None and len(examples) != expected_count:
        raise ValueError("unexpected split size")
    if not examples or len({x.id for x in examples}) != len(examples):
        raise ValueError("empty split or duplicate source IDs")
    if len({x.question for x in examples}) != len(examples):
        raise ValueError("duplicate questions in split")
    return examples


def load_split(data_root, split):
    name, count, digest = SPLITS[split]
    return load_jsonl(Path(data_root) / name, split, digest, count)


def audit_data(data_root):
    loaded = {split: load_split(data_root, split) for split in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if {x.question for x in loaded[a]} & {x.question for x in loaded[b]}:
            raise ValueError(f"question leakage between {a} and {b}")
    return {s: {"file": SPLITS[s][0], "count": len(rows), "sha256": SPLITS[s][2],
                "fields": ["id", "question", "cot", "answer"]} for s, rows in loaded.items()}


def question_prompt(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("a non-empty question is required")
    return f"Question: {question.strip()}\nReasoning:\n"


def progress_anchors(steps, count=2, max_chars=72):
    """Training labels at fixed progress positions, never a graph query."""
    if not steps or count == 0:
        return ()
    indices = sorted({min(len(steps) - 1, (j + 1) * len(steps) // (count + 1)) for j in range(count)})
    anchors = []
    for index in indices:
        equations = re.findall(r"<<([^<>]+)>>", steps[index])
        text = "; ".join(equations) if equations else " ".join(steps[index].split())
        anchors.append(text[:max_chars])
    return tuple(anchors)


def output_parts(example, config):
    anchors = progress_anchors(example.steps, config.anchor_count, config.anchor_max_chars)
    prefix = "\nAnchors:\n" if config.anchor_count else "\n"
    anchor_text = "\n".join(anchors) + "\n" if anchors else ""
    return prefix, anchor_text, f"Answer: {example.answer}"


def normalize_number(text):
    text = str(text).strip().replace(",", "").replace("$", "")
    try:
        value = Decimal(text)
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def answer_matches(completion, gold):
    """Require the explicit answer field; an anchor number is not an answer."""
    matches = re.findall(r"(?:^|\n)\s*Answer:\s*([^\n]*)", completion)
    if not matches:
        return False
    prediction = matches[-1].strip()
    number = re.fullmatch(r"\$?\s*([-+]?(?:\d[\d,]*\.?\d*|\.\d+))\s*\.?", prediction)
    predicted = normalize_number(number.group(1)) if number else None
    expected = normalize_number(gold)
    return predicted is not None and expected is not None and predicted == expected
