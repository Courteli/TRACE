import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl


FORMAL_GSM8K_DIR = Path(
    "/disk1/dingxukai/TRACE/data/raw/GSM8k-Aug-NL"
)
FORMAL_FILES = {
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
        # Arithmetic rationales often continue after units, for example
        # "8 ft. * 20 ft. = 160 sq. ft.". A capitalized next sentence is
        # still treated as a real boundary.
        return suffix[0].isupper()
    return True


def split_cot_steps(cot: str) -> List[str]:
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
            if not _is_cot_sentence_boundary(
                paragraph,
                punctuation_index,
            ):
                continue
            part = paragraph[start : punctuation_index + 1].strip()
            if part:
                steps.append(part)
            start = punctuation_index + 1
        tail = paragraph[start:].strip()
        if tail:
            steps.append(tail)
    return steps or [raw or "\n"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GSM8KAugNLProcessedDataset(Dataset):
    def __init__(self, data: List[dict]):
        super().__init__()
        self.data = {}
        for idx, item in enumerate(data):
            steps = split_cot_steps(item["cot"])
            row = {
                "idx": idx,
                "source_id": int(item.get("id", idx)),
                "question": item["question"],
                "steps": "\n".join(steps),
                "step_list_json": json.dumps(
                    steps,
                    ensure_ascii=False,
                ),
                "answer": str(item["answer"]),
                "n_steps": len(steps),
            }
            self.data[idx] = row
        self.all_indices = list(self.data.keys())
        self.indices = self.all_indices.copy()

    def get_all_indices(self):
        return self.all_indices

    def set_indices(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[self.indices[idx]]


class GSM8KAugNLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        dataset_dir: str,
        tiny_dataset: bool = False,
        epoch_scaling: int = 1,
        all_config=None,
        train_file: str = "gsm8k_train_processed.jsonl",
        val_file: str = "gsm8k_val_processed.jsonl",
        test_file: str = "gsm8k_test_processed.jsonl",
        enforce_registered_source: bool = False,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = Path(dataset_dir)
        self.tiny_dataset = tiny_dataset
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.enforce_registered_source = bool(enforce_registered_source)
        self.train_set: Optional[GSM8KAugNLProcessedDataset] = None
        self.val_set: Optional[GSM8KAugNLProcessedDataset] = None
        self.test_set: Optional[GSM8KAugNLProcessedDataset] = None

    def _load_jsonl(self, file_name: str) -> List[dict]:
        path = self.dataset_dir / file_name
        if self.enforce_registered_source:
            self._validate_registered_file(path)
        with path.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        ids = [int(item.get("id", index)) for index, item in enumerate(data)]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path} contains duplicate source ids")
        if self.tiny_dataset:
            data = data[:32]
        return data

    def _validate_registered_file(self, path: Path) -> None:
        if self.tiny_dataset:
            raise ValueError(
                "Formal TRACE source contract forbids tiny_dataset"
            )
        if self.dataset_dir.resolve() != FORMAL_GSM8K_DIR.resolve():
            raise ValueError(
                "Formal TRACE must read the immutable project-local mirror: "
                f"{FORMAL_GSM8K_DIR}"
            )
        if path.name not in FORMAL_FILES:
            raise ValueError(f"Unregistered formal GSM8K file: {path.name}")
        expected_count, expected_hash = FORMAL_FILES[path.name]
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{path} SHA256 {actual_hash} != {expected_hash}"
            )
        with path.open(encoding="utf-8") as handle:
            actual_count = sum(1 for line in handle if line.strip())
        if actual_count != expected_count:
            raise ValueError(
                f"{path} has {actual_count} rows, expected {expected_count}"
            )

    def _make_dataset(self, file_name: str) -> GSM8KAugNLProcessedDataset:
        data = self._load_jsonl(file_name)
        return GSM8KAugNLProcessedDataset(data)

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            self.train_set = self._make_dataset(self.train_file)
        if stage in (None, "fit", "validate"):
            self.val_set = self._make_dataset(self.val_file)
        if stage in (None, "test"):
            self.test_set = self._make_dataset(self.test_file)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.all_config.dataloader.batch_size,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
        )

    def get_dataloader_to_filter_indices(self):
        return DataLoader(self.train_set, batch_size=8, shuffle=False)

    def get_all_train_indices(self):
        return self.train_set.get_all_indices()

    def set_train_indices(self, train_indices):
        self.train_set.set_indices(train_indices)
