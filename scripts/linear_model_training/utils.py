from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def logistic_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Binary logistic loss (BCE with logits).

    Args:
        logits: [N] logits from LinearModel
        targets: [N] in {0, 1}

    Returns:
        scalar loss
    """
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def hinge_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Hinge loss (SVM-style).

    Args:
        logits: [N]
        targets: [N] in {0, 1} or {-1, +1}
    """
    unique = torch.unique(targets)
    vals = set(unique.tolist())
    if vals == {0, 1}:
        y = targets * 2 - 1   # 0 -> -1, 1 -> +1
    else:
        y = targets

    y = y.float()
    loss = torch.clamp(margin - y * logits, min=0.0)
    return loss.mean()


# ---------- Accuracy helper ----------

@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.0,
) -> float:
    """
    Compute accuracy for binary classification.

    Args:
        logits: [N]
        targets: [N] in {0, 1}
    """
    preds = (logits > threshold).long()
    targets = targets.long()
    return (preds == targets).float().mean().item()




def get_num_layers_and_dim(emb_path: Path):
    """
    Load embeddings once just to read shape, then delete.

    Returns:
        num_layers, dim_emb
    """
    obj = torch.load(emb_path, map_location="cpu")
    if "embeddings" not in obj:
        raise KeyError(f'"embeddings" not found in {emb_path}. Keys: {list(obj.keys())}')

    emb = obj["embeddings"]
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)

    if emb.dim() == 3:
        num_record, num_layer, dim_emb = emb.shape
    elif emb.dim() == 2:
        num_record, dim_emb = emb.shape
        num_layer = 1
    else:
        raise ValueError(f"Unexpected embedding shape {emb.shape}, expected [N,L,D] or [N,D].")

    # free memory
    del emb
    del obj
    return num_layer, dim_emb