import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset


def _gold_rationale(item: dict) -> dict:
    return {
        "steps": list(item["steps"]),
        "dependency_matrix": item.get("dependency_matrix"),
        "confidence_matrix": item.get(
            "confidence_matrix",
            item.get("dependency_confidence_matrix"),
        ),
        "fingerprint": item.get("rationale_fingerprint", "gold"),
        "source": item.get("rationale_source", "gold"),
        "verified": True,
    }


def normalize_rationale_set(item: dict) -> List[dict]:
    """Return a validated rationale set while preserving the gold rationale first."""
    gold = _gold_rationale(item)
    raw_rationales = item.get("rationale_set") or []
    normalized = [gold]
    seen = {str(gold["fingerprint"])}
    seen_steps = {json.dumps(gold["steps"], ensure_ascii=False)}

    for rationale in raw_rationales:
        if not isinstance(rationale, dict):
            continue
        steps = rationale.get("steps")
        if not isinstance(steps, list) or not steps:
            continue
        steps = [str(step).strip() for step in steps if str(step).strip()]
        if not steps:
            continue
        steps_key = json.dumps(steps, ensure_ascii=False)
        fingerprint = str(rationale.get("fingerprint", steps_key))
        if fingerprint in seen or steps_key in seen_steps:
            continue
        if rationale.get("verified") is not True:
            continue
        normalized.append(
            {
                "steps": steps,
                "dependency_matrix": rationale.get("dependency_matrix"),
                "confidence_matrix": rationale.get(
                    "confidence_matrix",
                    rationale.get("dependency_confidence_matrix"),
                ),
                "fingerprint": fingerprint,
                "source": str(rationale.get("source", "generated")),
                "verified": True,
            }
        )
        seen.add(fingerprint)
        seen_steps.add(steps_key)
    return normalized


class TraceRationaleSetDataset(Dataset):
    """QSA-compatible examples augmented with a verified set of correct rationales."""

    def __init__(self, data: List[dict]):
        super().__init__()
        self.data = {}
        for idx, item in enumerate(data):
            rationale_set = normalize_rationale_set(item)
            gold = rationale_set[0]
            anchor_indices = item.get("anchor_indices")
            anchor_bonus_indices = item.get(
                "anchor_bonus_indices",
                item.get("qwen_anchor_indices"),
            )
            self.data[idx] = {
                "idx": idx,
                "source_id": int(item.get("source_id", idx)),
                "question": item["question"],
                "answer": str(item["answer"]),
                "steps": "\n".join(gold["steps"]),
                "step_list_json": json.dumps(
                    gold["steps"],
                    ensure_ascii=False,
                ),
                "dependency_matrix_json": json.dumps(
                    gold.get("dependency_matrix")
                )
                if gold.get("dependency_matrix") is not None
                else "",
                "confidence_matrix_json": json.dumps(
                    gold.get("confidence_matrix")
                )
                if gold.get("confidence_matrix") is not None
                else "",
                "anchor_indices_json": json.dumps(anchor_indices)
                if anchor_indices is not None
                else "",
                "anchor_bonus_indices_json": json.dumps(anchor_bonus_indices)
                if anchor_bonus_indices is not None
                else "",
                "rationale_set_json": json.dumps(
                    rationale_set,
                    ensure_ascii=False,
                ),
                "n_rationales": len(rationale_set),
                "n_steps": len(gold["steps"]),
            }
        self.all_indices = list(self.data)
        self.indices = copy.deepcopy(self.all_indices)

    def get_all_indices(self):
        return self.all_indices

    def set_indices(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[self.indices[idx]]


class TraceRationaleSetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        tiny_dataset=False,
        epoch_scaling=1,
        dataset_dir: Optional[str] = None,
        all_config=None,
        **_,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = (
            Path(dataset_dir)
            if dataset_dir is not None
            else Path(
                all_config.args.workspace_path,
                "datasets",
                "text_reasoning",
                dataset_name,
            )
        )
        self.tiny_dataset = bool(tiny_dataset)
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def _load_split(self, split: str) -> List[dict]:
        path = self.dataset_dir / f"{split}.json"
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        if self.tiny_dataset:
            data = data[:32]
        return data

    def setup(self, stage: str = None):
        if stage in (None, "fit"):
            self.train_set = TraceRationaleSetDataset(
                self._load_split("train")
            )
            self.val_set = TraceRationaleSetDataset(self._load_split("val"))
        if stage in (None, "test"):
            self.test_set = TraceRationaleSetDataset(
                self._load_split("test")
            )

    def _loader(self, dataset, *, shuffle: bool, validation: bool = False):
        batch_size = (
            self.all_config.dataloader.get("val_batch_size", 1)
            if validation
            else self.all_config.dataloader.batch_size
        )
        num_workers = self.all_config.dataloader.get("num_workers", 4)
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
            persistent_workers=(
                self.all_config.dataloader.get("persistent_workers", True)
                and num_workers > 0
            ),
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(
            self.val_set,
            shuffle=False,
            validation=True,
        )

    def test_dataloader(self):
        return self._loader(
            self.test_set,
            shuffle=False,
            validation=True,
        )

    def get_dataloader_to_filter_indices(self):
        return DataLoader(self.train_set, batch_size=8, shuffle=False)

    def get_all_train_indices(self):
        return self.train_set.get_all_indices()

    def set_train_indices(self, train_indices):
        self.train_set.set_indices(train_indices)
