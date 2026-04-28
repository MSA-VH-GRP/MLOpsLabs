"""
PyTorch Dataset classes for Mamba4Rec sequential recommendation.

Data is consumed from the list-of-dicts format produced by
src/training/mamba_trainer.py::build_sequences_and_split().

Each sample dict has the structure:
    {
        "item_seq":    List[int],             # ordered watched items (input)
        "time_seq":    List[int],             # time-slot per item
        "genre_seq":   List[List[int]],       # genre indices per item (padded to max_genres)
        "user_profile": {
            "age_idx":    int,
            "gender_idx": int,
            "occupation": int,
        },
        "target":      int,                   # next item (target for training)
        "target_time": int,                   # time-slot of the target event
    }

EvalDataset adds a "candidates" field: [target] + 99 randomly-sampled negatives.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple


class SequentialDataset(Dataset):
    """Training dataset for next-item prediction."""

    def __init__(
        self,
        data: List[Dict],
        max_seq_len: int = 50,
        max_genres: int = 3,
    ):
        self.data = data
        self.max_seq_len = max_seq_len
        self.max_genres = max_genres

    def __len__(self) -> int:
        return len(self.data)

    def _pad_sequence(self, seq: List, max_len: int, pad_value: int = 0) -> List:
        if len(seq) >= max_len:
            return seq[-max_len:]
        return [pad_value] * (max_len - len(seq)) + seq

    def _pad_genre_sequence(self, genre_seq: List[List[int]]) -> List[List[int]]:
        padded = []
        for genres in genre_seq:
            if len(genres) < self.max_genres:
                genres = genres + [0] * (self.max_genres - len(genres))
            else:
                genres = genres[: self.max_genres]
            padded.append(genres)

        while len(padded) < self.max_seq_len:
            padded.insert(0, [0] * self.max_genres)

        return padded[-self.max_seq_len :]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        item_seq = self._pad_sequence(sample["item_seq"], self.max_seq_len)
        time_seq = self._pad_sequence(sample["time_seq"], self.max_seq_len)
        genre_seq = self._pad_genre_sequence(sample["genre_seq"])

        up = sample["user_profile"]
        return {
            "item_seq":    torch.LongTensor(item_seq),
            "genre_seq":   torch.LongTensor(genre_seq),
            "time_seq":    torch.LongTensor(time_seq),
            "delta_seq":   torch.LongTensor(self._pad_sequence(sample.get("delta_seq", []), self.max_seq_len)),
            "age_idx":     torch.LongTensor([up["age_idx"]]),
            "gender_idx":  torch.LongTensor([up["gender_idx"]]),
            "occupation":  torch.LongTensor([up["occupation"]]),
            "target":      torch.LongTensor([sample["target"]]),
            "target_time": torch.LongTensor([sample["target_time"]]),
        }


class EvalDataset(Dataset):
    """Evaluation dataset with 99 negative samples per positive."""

    def __init__(
        self,
        data: List[Dict],
        num_items: int,
        num_negatives: int = 99,
        max_seq_len: int = 50,
        max_genres: int = 3,
    ):
        self.data = data
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len
        self.max_genres = max_genres
        self._prepare_eval_data()

    def _prepare_eval_data(self):
        self.eval_data = []
        for sample in self.data:
            history = set(sample["item_seq"])
            target = sample["target"]

            negatives: List[int] = []
            while len(negatives) < self.num_negatives:
                neg = np.random.randint(1, self.num_items)
                if neg not in history and neg != target:
                    negatives.append(neg)

            self.eval_data.append({**sample, "candidates": [target] + negatives})

    def _pad_sequence(self, seq: List, max_len: int, pad_value: int = 0) -> List:
        if len(seq) >= max_len:
            return seq[-max_len:]
        return [pad_value] * (max_len - len(seq)) + seq

    def _pad_genre_sequence(self, genre_seq: List[List[int]]) -> List[List[int]]:
        padded = []
        for genres in genre_seq:
            if len(genres) < self.max_genres:
                genres = genres + [0] * (self.max_genres - len(genres))
            else:
                genres = genres[: self.max_genres]
            padded.append(genres)

        while len(padded) < self.max_seq_len:
            padded.insert(0, [0] * self.max_genres)

        return padded[-self.max_seq_len :]

    def __len__(self) -> int:
        return len(self.eval_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.eval_data[idx]

        item_seq = self._pad_sequence(sample["item_seq"], self.max_seq_len)
        time_seq = self._pad_sequence(sample["time_seq"], self.max_seq_len)
        genre_seq = self._pad_genre_sequence(sample["genre_seq"])

        up = sample["user_profile"]
        return {
            "item_seq":    torch.LongTensor(item_seq),
            "genre_seq":   torch.LongTensor(genre_seq),
            "time_seq":    torch.LongTensor(time_seq),
            "delta_seq":   torch.LongTensor(self._pad_sequence(sample.get("delta_seq", []), self.max_seq_len)),
            "age_idx":     torch.LongTensor([up["age_idx"]]),
            "gender_idx":  torch.LongTensor([up["gender_idx"]]),
            "occupation":  torch.LongTensor([up["occupation"]]),
            "candidates":  torch.LongTensor(sample["candidates"]),
            "target_time": torch.LongTensor([sample["target_time"]]),
        }


def create_dataloaders(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    metadata: Dict,
    batch_size: int = 256,
    num_workers: int = 0,
    max_seq_len: int = 50,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders."""
    max_genres = metadata.get("max_genres_per_item", 3)
    num_items = metadata["num_items"]

    train_dataset = SequentialDataset(train_data, max_seq_len=max_seq_len, max_genres=max_genres)
    val_dataset = EvalDataset(val_data, num_items=num_items, max_seq_len=max_seq_len, max_genres=max_genres)
    test_dataset = EvalDataset(test_data, num_items=num_items, max_seq_len=max_seq_len, max_genres=max_genres)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader
