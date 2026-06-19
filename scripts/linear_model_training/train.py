
import argparse
from pathlib import Path

import torch
import os
import random
import numpy as np

from dataset import make_dataloaders_from_files
from trainer import train_with_early_stopping
from utils import logistic_loss, hinge_loss, compute_accuracy, get_num_layers_and_dim

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)



def main():
    parser = argparse.ArgumentParser(description="Layer-wise linear training with early stopping.")
    parser.add_argument("--emb_dir", type=str, required=True,
                        help="Folder containing embedding .pt file.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Folder containing benchmark CSV file.")
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Benchmark name (used in file naming).")
    parser.add_argument("--column", type=str, required=True,
                        help="Column name used in embedding file naming.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name used in embedding file naming.")

    parser.add_argument("--loss_type", type=str, default="logistic",
                        choices=["logistic", "hinge"],
                        help="Which loss to use: logistic (log-reg) or hinge (SVM-style).")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate for SGD.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for SGD.")
    parser.add_argument("--max_epochs", type=int, default=1000,
                        help="Maximum number of epochs per layer.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience on dev.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for all splits.")

    args = parser.parse_args()

    out_folder = args.loss_type
    os.makedirs(out_folder, exist_ok=True)

    emb_path = Path(args.emb_dir) / f"{args.benchmark}_{args.column}_{args.model}.pt"
    csv_path = Path(args.dataset_dir) / f"{args.benchmark}.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    print(f"Embedding file: {emb_path}")
    print(f"Dataset file:   {csv_path}")

    # 1. Read number of layers and embedding dim, then drop embeddings
    num_layers, dim_emb = get_num_layers_and_dim(emb_path)
    print(f"Found {num_layers} layer(s), embedding dim = {dim_emb}")

    # 2. Allocate containers for results
    W = torch.zeros(num_layers, dim_emb, dtype=torch.float32)
    b = torch.zeros(num_layers, dtype=torch.float32)

    dev_loss = torch.zeros(num_layers, dtype=torch.float32)
    dev_acc = torch.zeros(num_layers, dtype=torch.float32)
    test_loss = torch.zeros(num_layers, dtype=torch.float32)
    test_acc = torch.zeros(num_layers, dtype=torch.float32)

    # 3. Train layer by layer
    for layer_idx in range(num_layers):
        print(f"\n=== Layer {layer_idx} / {num_layers - 1} ===")

        loaders = make_dataloaders_from_files(
            emb_path=emb_path,
            csv_path=csv_path,
            layer=layer_idx if num_layers > 1 else 0,
            batch_size=args.batch_size,
            shuffle_train=True,
        )

        model, history, metrics = train_with_early_stopping(
            loaders=loaders,
            loss_type=args.loss_type,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            use_cuda_if_available=True,
            early_stopping_metric="dev_loss",
        )

        # Extract weight (and bias if present) from the final/best model
        lin = model.linear
        weight = lin.weight.detach().cpu().view(-1)  # [D]
        W[layer_idx] = weight

        if hasattr(lin, "bias") and lin.bias is not None:
            b[layer_idx] = lin.bias.detach().cpu().item()
        else:
            b[layer_idx] = 0.0  # no-bias model: store 0 for convenience

        dev_loss[layer_idx] = metrics["dev_loss"]
        dev_acc[layer_idx] = metrics["dev_acc"]
        test_loss[layer_idx] = metrics["test_loss"]
        test_acc[layer_idx] = metrics["test_acc"]

        print(
            f"Layer {layer_idx}: dev_loss={dev_loss[layer_idx]:.4f}, "
            f"dev_acc={dev_acc[layer_idx]:.4f}, "
            f"test_loss={test_loss[layer_idx]:.4f}, "
            f"test_acc={test_acc[layer_idx]:.4f}"
        )

    # 4. Build result dict
    result = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "W": W,                 # [num_layers, dim_emb]
        "b": b,                 # [num_layers]
        "dev_loss": dev_loss,   # [num_layers]
        "dev_acc": dev_acc,     # [num_layers]
        "test_loss": test_loss, # [num_layers]
        "test_acc": test_acc,   # [num_layers]
        "num_layers": num_layers,
        "dim_emb": dim_emb,
        "benchmark": args.benchmark,
        "column": args.column,
        "model": args.model,
        "embedding_file": str(emb_path),
        "dataset_file": str(csv_path),
    }


    out_path = os.path.join(out_folder, f"{args.benchmark}_{args.column}_{args.model}_linear_{args.loss_type}.pt")


    torch.save(result, out_path)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
