# rq-representation-probing

**Official code for:**  
*Rhetorical Questions in LLM Representations: A Linear Probing Study*  
**Authors:** Louie Hong Yao, Vishesh Anand, Yuan Zhuang, Tianyu Jiang

---

## Overview

Understanding how large language models represent rhetorical questions is important for studying non-literal language and communicative intent.

This project investigates how rhetorical-question intent is encoded in LLM representations using linear probing. We study whether rhetorical questions are linearly separable from information-seeking questions, how this signal evolves across layers, how different linear probes compare, and how probing directions transfer across datasets.

More broadly, this work asks whether strong transfer performance implies a shared representation, or whether different probes can achieve similar discrimination while relying on distinct representational directions.

Current focus of the project includes:

- analyzing rhetorical-question separability across layers,
- comparing diffMean, logistic, and hinge-based linear probes,
- measuring alignment between probing directions and induced rankings,
- studying cross-dataset transfer of rhetorical signals,
- examining the heterogeneity of rhetorical cues in LLM representations.

## Data

The repository includes the RQ dataset at `data/RQ.csv`. The file contains context-question pairs with labels and dataset split metadata used for the rhetorical-question analyses.

The SRAQ data is not included in this repository. It should be obtained from the source associated with the SRAQ paper and placed in the local data directory before running analyses that depend on SRAQ.

## Code

The analysis code is organized under `scritps/`:

- `scritps/diffmean_analysis.ipynb`  
  Runs the plain diffMean analysis on model representations.
- `scritps/pca_transform.py`  
  Applies layer-wise PCA to embedding `.pt` files and saves both compressed embeddings and PCA bases.
- `scritps/linear_models/`  
  Contains notebooks for analyses in the original representation space, including within-dataset cosine similarity, Spearman correlation, Jaccard index, projection AUROC, and cross-dataset projection transfer.
- `scritps/pca_map_back/`  
  Contains notebooks for PCA-space analyses and mapping directions back to the original representation space, including projection transfer, projection AUROC, and Spearman/Jaccard comparisons.

### PCA transformation

To create PCA-compressed embeddings for a directory of `.pt` files:

```bash
python scritps/pca_transform.py --emb_dir /path/to/embeddings --n_components 64
```

Each input file is expected to contain an `embeddings` tensor with shape `(N, L, D)`. The script writes compressed embeddings to a sibling directory named `{emb_dir}_pca_{n_components}` and saves the corresponding PCA components, means, and explained-variance ratios under its `pca_basis/` subdirectory.

### Analysis notebooks

The notebooks assume precomputed embedding files and, for PCA map-back analyses, PCA basis files produced by `scritps/pca_transform.py`. Run the notebooks after updating any local data paths to point to the relevant embedding and PCA-output directories.
