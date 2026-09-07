if __name__ == "__main__":
    import sys

    sys.path.append("../../")

import copy
from pathlib import Path
import json
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class QuestionStepsAnswerDataset(Dataset):
    def __init__(
        self,
        data: List[dict],
    ):
        super().__init__()
        self.data = {}
        for idx, d in enumerate(data):
            dependency_matrix = d.get("dependency_matrix")
            confidence_matrix = d.get("confidence_matrix", d.get("dependency_confidence_matrix"))
            anchor_indices = d.get("anchor_indices")
            anchor_bonus_indices = d.get("anchor_bonus_indices", d.get("qwen_anchor_indices"))
            self.data[idx] = {
                "idx": idx,
                "question": d["question"],
                "answer": d["answer"],
                "steps": "\n".join(d["steps"]),
                "step_list_json": json.dumps(d["steps"], ensure_ascii=False),
                "dependency_matrix_json": json.dumps(dependency_matrix) if dependency_matrix is not None else "",
                "confidence_matrix_json": json.dumps(confidence_matrix) if confidence_matrix is not None else "",
                "anchor_indices_json": json.dumps(anchor_indices) if anchor_indices is not None else "",
                "anchor_bonus_indices_json": json.dumps(anchor_bonus_indices) if anchor_bonus_indices is not None else "",
                "n_steps": len(d["steps"]),
            }
        self.all_indices = list(self.data.keys())
        self.indices = copy.deepcopy(self.all_indices)

    def get_all_indices(self):
        return self.all_indices

    def set_indices(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        data_idx = self.indices[idx]
        return self.data[data_idx]


class QSADataModule(pl.LightningDataModule):
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
        if dataset_dir is not None:
            self.dataset_dir = Path(dataset_dir)
        else:
            self.dataset_dir = Path(all_config.args.workspace_path, "datasets", "text_reasoning", dataset_name)
        self.tiny_dataset = tiny_dataset
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config
        self.batch_size = all_config.dataloader.batch_size

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str = None):
        def load_split(split: str):
            with open(self.dataset_dir / f"{split}.json") as f:
                data = json.load(f)
                if self.tiny_dataset:
                    data = data[:32]
            return data

        # Initialize datasets
        if stage == "fit":
            self.train_set = self._create_dataset(load_split("train"), "train")
            self.val_set = self._create_dataset(load_split("val"), "val")
        elif stage == "test":
            self.test_set = self._create_dataset(load_split("test"), "test")

    def _create_dataset(self, raw_data: List[dict], mode: str) -> QuestionStepsAnswerDataset:
        return QuestionStepsAnswerDataset(
            data=raw_data,
        )

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
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.all_config.dataloader.get("val_batch_size", 1),
            shuffle=False,
            num_workers=self.all_config.dataloader.get("num_workers", 4),
            persistent_workers=self.all_config.dataloader.get("persistent_workers", True),
            pin_memory=self.all_config.dataloader.get("pin_memory", True),
        )

    def get_dataloader_to_filter_indices(self):
        return DataLoader(
            self.train_set,
            batch_size=8,
            shuffle=False,
        )

    def get_all_train_indices(self):
        return self.train_set.get_all_indices()

    def set_train_indices(self, train_indices):
        self.train_set.set_indices(train_indices)
