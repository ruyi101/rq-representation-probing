# trainer.py

from typing import Dict, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from model import LinearModel
from utils import logistic_loss, hinge_loss, compute_accuracy


def _eval_loop(
    model: LinearModel,
    loader: DataLoader,
    loss_type: Literal["logistic", "hinge"],
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model on a given loader.

    Returns:
        avg_loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    if len(loader.dataset) == 0:
        return float("nan"), float("nan")

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            if loss_type == "logistic":
                loss = logistic_loss(logits, y)
            elif loss_type == "hinge":
                loss = hinge_loss(logits, y)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            acc_batch = compute_accuracy(logits, y)
            total_correct += int(acc_batch * batch_size)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def train_with_early_stopping(
    loaders: Dict[str, DataLoader],
    loss_type: Literal["logistic", "hinge"] = "logistic",
    lr: float = 0.1,
    weight_decay: float = 0.0,
    max_epochs: int = 1000,
    patience: int = 20,
    use_cuda_if_available: bool = True,
    early_stopping_metric: Literal["dev_loss", "dev_acc"] = "dev_loss",
):
    """
    Train a LinearModel on train, monitor on dev, early-stop, then evaluate on dev/test.

    Args:
        loaders: dict with keys "train", "dev", "test" -> DataLoader
        loss_type: "logistic" or "hinge"
        lr: learning rate for SGD
        weight_decay: weight decay for SGD
        max_epochs: maximum number of epochs
        patience: stop if no improvement on dev for this many epochs
        use_cuda_if_available: use GPU if available
        early_stopping_metric: "dev_loss" (minimize) or "dev_acc" (maximize)

    Returns:
        model: trained LinearModel (restored to best dev point)
        history: dict with lists of train/dev metrics
        final_metrics: dict with dev/test loss/acc at best checkpoint
    """
    device = torch.device("cuda" if (use_cuda_if_available and torch.cuda.is_available()) else "cpu")

    train_loader = loaders["train"]
    dev_loader = loaders["validation"]
    test_loader = loaders["test"]

    if len(train_loader.dataset) == 0:
        raise ValueError("Train split is empty — cannot train.")

    if len(dev_loader.dataset) == 0:
        raise ValueError("Dev split is empty — cannot early-stop without dev set.")
    
    print(f"Traing samples: {len(train_loader.dataset)} | Dev samples: {len(dev_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # Infer input dim from one batch
    X_batch, y_batch = next(iter(train_loader))
    in_dim = X_batch.shape[1]

    model = LinearModel(in_dim=in_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
    }

    # Early-stopping state
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    if early_stopping_metric == "dev_loss":
        best_score = float("inf")
        mode = "min"
    elif early_stopping_metric == "dev_acc":
        best_score = -float("inf")
        mode = "max"
    else:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")

    for epoch in range(1, max_epochs + 1):
        # ---------- training ----------
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)

            if loss_type == "logistic":
                loss = logistic_loss(logits, y)
            elif loss_type == "hinge":
                loss = hinge_loss(logits, y)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            total_correct += int(compute_accuracy(logits, y) * batch_size)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        # ---------- dev evaluation ----------
        dev_loss, dev_acc = _eval_loop(model, dev_loader, loss_type, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["dev_loss"].append(dev_loss)
        history["dev_acc"].append(dev_acc)

        # print(
        #     f"Epoch {epoch:4d} | "
        #     f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
        #     f"dev_loss={dev_loss:.4f}, dev_acc={dev_acc:.4f}"
        # )

        # ---------- early stopping check ----------
        if early_stopping_metric == "dev_loss":
            score = dev_loss
        else:  # "dev_acc"
            score = dev_acc

        improved = (mode == "min" and score < best_score) or (mode == "max" and score > best_score)

        if improved:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    # ---------- restore best model ----------
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    else:
        print("Warning: no improvement recorded; using last-epoch model.")

    # ---------- final dev / test evaluation ----------
    dev_loss, dev_acc = _eval_loop(model, dev_loader, loss_type, device)
    test_loss, test_acc = _eval_loop(model, test_loader, loss_type, device)

    final_metrics = {
        "dev_loss": dev_loss,
        "dev_acc": dev_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }

    print(
        f"\nFinal (best) dev:  loss={dev_loss:.4f}, acc={dev_acc:.4f}\n"
        f"Final (best) test: loss={test_loss:.4f}, acc={test_acc:.4f}"
    )

    return model, history, final_metrics
