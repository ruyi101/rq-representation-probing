#!/usr/bin/env python
"""
Given a folder of .pt files with embeddings of shape (N, L, D),
for each file:

    1. For each layer ℓ in {0,...,L-1}, fit PCA with 256 components
       on the embeddings[:, ℓ, :].
    2. Save compressed embeddings of shape (N, L, 256) to a new
       folder: {orig_folder}_pca, with the same filename.
    3. In {orig_folder}_pca/pca_basis/, save the PCA bases for that
       file as a tensor of shape (L, 256, D) along with metadata.

All original keys 'model_name', 'benchmark', 'column' are preserved.
"""

import argparse
from pathlib import Path

import torch
from sklearn.decomposition import PCA


def process_file(path: Path, out_root: Path, basis_root: Path, n_components: int = 256):
    print(f"\nProcessing: {path}")

    obj = torch.load(path, map_location="cpu")

    if "embeddings" not in obj:
        print(f"  Skipping (no 'embeddings' key): {path}")
        return

    emb = obj["embeddings"]
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)

    if emb.dim() != 3:
        raise ValueError(
            f"Expected embeddings to have shape (N, L, D), got {emb.shape} in {path}"
        )

    N, L, D = emb.shape
    print(f"  embeddings shape: N={N}, L={L}, D={D}")

    # Allocate compressed embeddings and basis
    n_comp = min(n_components, D, N)  # safe, though you likely have N,D >= 256
    print(f"  Using n_components={n_comp}")

    emb_pca = torch.empty(N, L, n_comp, dtype=torch.float32)
    basis = torch.empty(L, n_comp, D, dtype=torch.float32)
    means = torch.empty(L, D, dtype=torch.float32)
    variance_ratio = torch.empty(L, n_comp, dtype=torch.float32)

    # Compute PCA layer-by-layer
    for layer_idx in range(L):
        print(f"    Layer {layer_idx+1}/{L} ...", end="", flush=True)

        X_layer = emb[:, layer_idx, :]  # [N, D]
        X_layer = X_layer.to(torch.float32).cpu()
        X_np = X_layer.numpy()

        pca = PCA(n_components=n_comp)
        X_trans = pca.fit_transform(X_np)       # [N, n_comp]
        comps = pca.components_                # [n_comp, D]
        mean = pca.mean_
        
        variance_ratio[layer_idx] = torch.from_numpy(pca.explained_variance_ratio_).to(torch.float32)
        emb_pca[:, layer_idx, :] = torch.from_numpy(X_trans).to(torch.float32)
        basis[layer_idx, :, :] = torch.from_numpy(comps).to(torch.float32)
        means[layer_idx, :] = torch.from_numpy(mean).to(torch.float32)

        print(" done.")

    # Build output file paths
    out_root.mkdir(parents=True, exist_ok=True)
    basis_root.mkdir(parents=True, exist_ok=True)

    out_file = out_root / path.name
    basis_file = basis_root / f"{path.stem}_pca_basis.pt"

    # Save compressed embeddings (keep other metadata)
    new_obj = dict(obj)
    new_obj["embeddings"] = emb_pca
    new_obj["pca_dim"] = n_comp
    new_obj["orig_dim"] = D

    torch.save(new_obj, out_file)
    print(f"  Saved compressed embeddings to: {out_file}")

    # Save PCA basis (L, n_comp, D) + some metadata
    basis_obj = {
        "pca_components": basis,   # [L, n_comp, D]
        "pca_means": means,        # [L, D]
        "n_components": n_comp,
        "explained_variance_ratio": variance_ratio,  # [L]
        "orig_dim": D,
        "num_layers": L,
    }
    # carry over some metadata if present
    for key in ["model_name", "benchmark", "column"]:
        if key in obj:
            basis_obj[key] = obj[key]

    torch.save(basis_obj, basis_file)
    print(f"  Saved PCA basis to: {basis_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply PCA per layer to all .pt embedding files in a folder."
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        required=True,
        help="Folder containing .pt files with 'embeddings' of shape (N, L, D).",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=64,
        help="Number of PCA components (default: 64).",
    )

    args = parser.parse_args()
    emb_dir = Path(args.emb_dir).expanduser().resolve()

    if not emb_dir.is_dir():
        raise NotADirectoryError(f"{emb_dir} is not a directory.")

    # New root folder: {old}_pca
    out_root = emb_dir.parent / f"{emb_dir.name}_pca_{args.n_components}"
    basis_root = out_root / "pca_basis"

    print(f"Input folder: {emb_dir}")
    print(f"Output folder for compressed embeddings: {out_root}")
    print(f"Output folder for PCA bases: {basis_root}")

    pt_files = sorted(emb_dir.glob("*.pt"))
    if not pt_files:
        print("No .pt files found in the given directory.")
        return

    for f in pt_files:
        temp = torch.load(f, map_location="cpu")
        model = temp['model_name']
        model = model.split('/')[-1]  # get the last part if there's a path
        # if model == 'Llama-3.3-70B-Instruct':
        #     n_components = 256
        #     print(f"Using n_components={n_components} for model {model}")
        # elif model in ['Llama-3.1-8B-Instruct', 'Qwen3-32B']:
        #     n_components = 128
        #     print(f"Using n_components={n_components} for model {model}")
        # elif model in ['Qwen3-8B', 'gpt-oss-20b']:
        #     n_components = 64
        #     print(f"Using n_components={n_components} for model {model}")
        # else:
        #     raise ValueError(f"Unknown model name: {model}")
        process_file(f, out_root=out_root, basis_root=basis_root, n_components=args.n_components)

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
