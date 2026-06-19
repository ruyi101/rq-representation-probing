# data_linear.py

from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EmbeddingDataset(Dataset):
    """
    Simple dataset wrapping (X, y).

    X: [N, D] float tensor
    y: [N] long tensor in {0, 1}
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.shape[0] == y.shape[0], f"X and y must have same length, got {X.shape[0]} vs {y.shape[0]}"
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _load_embeddings_for_layer(
    emb_path: Union[str, Path],
    layer: int,
) -> torch.Tensor:
    """
    Load embedding tensor from .pt and select a single layer.

    emb_path: path to .pt file with key "embeddings"
              shape (num_record, num_layer, dim_emb)

    layer: index of layer to use (0-based; can be negative like -1)

    Returns:
        X_all: [num_record, dim_emb] float32 tensor
    """
    emb_path = Path(emb_path)
    obj = torch.load(emb_path, map_location="cpu")

    if "embeddings" not in obj:
        raise KeyError(f'"embeddings" not found in {emb_path}. Keys: {list(obj.keys())}')

    emb = obj["embeddings"]  # expect [N, L, D] or [N, D]

    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)

    if emb.dim() == 2:
        # already [N, D]
        X_all = emb.float()
    elif emb.dim() == 3:
        # [N, L, D]
        N, L, D = emb.shape
        if not (-L <= layer < L):
            raise IndexError(f"Layer index {layer} out of range for num_layer={L}")
        X_all = emb[:, layer, :].float()  # [N, D]
    else:
        raise ValueError(f"Unexpected embedding shape {emb.shape}, expected 2D or 3D.")

    return X_all


def _load_split_indices_and_labels(
    csv_path: Union[str, Path],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Read the CSV and extract indices + labels per split.

    CSV columns:
        - id           : which row in embeddings
        - binary_label : 0/1
        - dataset      : 'train', 'dev', or 'test'

    Returns:
        splits: dict[str, (idx_tensor, y_tensor)]
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {"id", "binary_label", "dataset"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {missing} in {csv_path}")

    splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for split_name in ["train", "validation", "test"]:
        df_split = df[df["dataset"] == split_name]

        if df_split.empty:
            # you can decide whether to error or just skip;
            # here we create empty tensors
            idx = torch.empty(0, dtype=torch.long)
            y = torch.empty(0, dtype=torch.long)
        else:
            idx = torch.tensor(df_split["id"].to_numpy(), dtype=torch.long)
            y = torch.tensor(df_split["binary_label"].to_numpy(), dtype=torch.long)

        splits[split_name] = (idx, y)

    return splits


def make_dataloaders_from_files(
    emb_path: Union[str, Path],
    csv_path: Union[str, Path],
    layer: int,
    batch_size: int = 128,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """
    Main entry point.

    Args:
        emb_path: .pt file with embeddings dict
        csv_path: .csv file with columns id, binary_label, dataset
        layer: which layer index to use (0-based; can be negative for PyTorch-style)
        batch_size: batch size for all splits
        shuffle_train: whether to shuffle the train split

    Returns:
        loaders: dict with keys 'train', 'dev', 'test' -> DataLoader
    """
    # (1) Load embeddings for desired layer -> [N, D]
    X_all = _load_embeddings_for_layer(emb_path, layer=layer)

    # (2) Read csv -> indices + labels per split
    split_info = _load_split_indices_and_labels(csv_path)

    loaders: Dict[str, DataLoader] = {}

    for split_name in ["train", "validation", "test"]:
        idx, y = split_info[split_name]

        if idx.numel() == 0:
            # Empty split -> empty loader
            dataset = EmbeddingDataset(
                X=torch.empty(0, X_all.shape[1], dtype=torch.float32),
                y=torch.empty(0, dtype=torch.long),
            )
        else:
            # sanity check: indices in range
            if idx.max().item() >= X_all.shape[0] or idx.min().item() < 0:
                raise IndexError(
                    f"Indices for split '{split_name}' out of range. "
                    f"Got [{idx.min().item()}, {idx.max().item()}], "
                    f"but embeddings have N={X_all.shape[0]}."
                )
            X_split = X_all[idx]  # [n_split, D]
            dataset = EmbeddingDataset(X_split, y)

        if split_name == "train":
            shuffle = shuffle_train
        else:
            shuffle = False

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return loaders
