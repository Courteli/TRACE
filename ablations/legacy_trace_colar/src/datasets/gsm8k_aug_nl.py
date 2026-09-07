import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl


class GSM8KAugNLProcessedDataset(Dataset):
    def __init__(self, data: List[dict], trace_teacher_paths: Optional[torch.Tensor] = None):
        super().__init__()
        self.data = {}
        if trace_teacher_paths is not None and len(trace_teacher_paths) < len(data):
            raise ValueError(
                f"trace_teacher_paths has {len(trace_teacher_paths)} rows, but dataset has {len(data)} items"
            )
        for idx, item in enumerate(data):
            row = {
                "idx": idx,
                "source_id": item.get("id", idx),
                "question": item["question"],
                "steps": item["cot"],
                "answer": str(item["answer"]),
            }
            if trace_teacher_paths is not None:
                row["trace_teacher_path"] = trace_teacher_paths[idx]
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
        trace_teacher_cache_dir: Optional[str] = None,
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
        self.trace_teacher_cache_dir = Path(trace_teacher_cache_dir) if trace_teacher_cache_dir else None
        self.train_set: Optional[GSM8KAugNLProcessedDataset] = None
        self.val_set: Optional[GSM8KAugNLProcessedDataset] = None
        self.test_set: Optional[GSM8KAugNLProcessedDataset] = None

    def _load_jsonl(self, file_name: str) -> List[dict]:
        path = self.dataset_dir / file_name
        with path.open("r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        if self.tiny_dataset:
            data = data[:32]
        return data

    def _trace_steps(self) -> Optional[int]:
        try:
            return int(self.all_config.model.model_kwargs.trace_trajectory_config.trace_steps)
        except Exception:
            return None

    def _load_trace_teacher_cache(self, file_name: str, expected_len: int) -> Optional[torch.Tensor]:
        if self.trace_teacher_cache_dir is None:
            return None
        trace_steps = self._trace_steps()
        if trace_steps is None:
            return None
        stem = Path(file_name).stem
        candidates = []
        if self.tiny_dataset:
            candidates.append(self.trace_teacher_cache_dir / f"{stem}_trace_teacher_k{trace_steps}_tiny.pt")
        candidates.append(self.trace_teacher_cache_dir / f"{stem}_trace_teacher_k{trace_steps}.pt")
        cache_path = next((path for path in candidates if path.exists()), candidates[-1])
        if not cache_path.exists():
            raise FileNotFoundError(
                f"TRACE teacher cache not found. Tried: {[str(path) for path in candidates]}. "
                "Build it with tools/trace_build_teacher_cache.py or unset trace_teacher_cache_dir."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            paths = payload.get("teacher_paths")
            if paths is None:
                paths = payload.get("paths")
        else:
            paths = payload
        if paths is None:
            raise ValueError(f"TRACE teacher cache has no teacher_paths tensor: {cache_path}")
        if len(paths) < expected_len:
            raise ValueError(
                f"TRACE teacher cache {cache_path} has {len(paths)} rows, expected at least {expected_len}"
            )
        return paths[:expected_len]

    def _make_dataset(self, file_name: str) -> GSM8KAugNLProcessedDataset:
        data = self._load_jsonl(file_name)
        trace_teacher_paths = self._load_trace_teacher_cache(file_name=file_name, expected_len=len(data))
        return GSM8KAugNLProcessedDataset(data, trace_teacher_paths=trace_teacher_paths)

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_set = self._make_dataset(self.train_file)
            self.val_set = self._make_dataset(self.val_file)
        elif stage == "test":
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
