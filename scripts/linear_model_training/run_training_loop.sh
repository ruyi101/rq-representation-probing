#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 BENCHMARK COLUMN LOSS_TYPE"
  echo "  BENCHMARK : e.g. SRAQ, RQ"
  echo "  COLUMN    : e.g. paragraph, question"
  echo "  LOSS_TYPE : logistic or hinge"
  exit 1
fi

BENCHMARK="$1"
COLUMN="$2"
LOSS_TYPE="$3"

# ---------- paths (edit as needed) ----------
EMB_DIR="../embeddings_mean_pca_64"        # folder containing *.pt embeddings
DATASET_DIR="../"      # folder containing benchmark CSVs

# ---------- defaults (tweak here) ----------
LR=1e-5
WEIGHT_DECAY=1e-4
MAX_EPOCHS=1000
PATIENCE=100
BATCH_SIZE=256

# Optional: adjust defaults by loss_type if you ever want
case "${LOSS_TYPE}" in
  logistic)
    # keep defaults
    ;;
  hinge)
    # you could tweak here if desired, e.g.:
    # LR=0.01
    # WEIGHT_DECAY=1e-2
    ;;
  *)
    echo "ERROR: LOSS_TYPE must be 'logistic' or 'hinge', got '${LOSS_TYPE}'"
    exit 1
    ;;
esac

# ---------- models to loop over ----------
MODELS=(
  "gpt-oss-20b"
  "Qwen3-8B"
  "Qwen3-32B"
  # "Llama-3.1-8B-Instruct"
  # "Llama-3.3-70B-Instruct"
)

echo "Running linear probe for:"
echo "  BENCHMARK    = ${BENCHMARK}"
echo "  COLUMN       = ${COLUMN}"
echo "  LOSS_TYPE    = ${LOSS_TYPE}"
echo "  EMB_DIR      = ${EMB_DIR}"
echo "  DATASET_DIR  = ${DATASET_DIR}"
echo "  LR           = ${LR}"
echo "  WEIGHT_DECAY = ${WEIGHT_DECAY}"
echo "  MAX_EPOCHS   = ${MAX_EPOCHS}"
echo "  PATIENCE     = ${PATIENCE}"
echo "  BATCH_SIZE   = ${BATCH_SIZE}"
echo

for MODEL in "${MODELS[@]}"; do
  echo "=== Model: ${MODEL} ==="

  python train.py \
    --emb_dir "${EMB_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    --benchmark "${BENCHMARK}" \
    --column "${COLUMN}" \
    --model "${MODEL}" \
    --loss_type "${LOSS_TYPE}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_epochs "${MAX_EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}"

  echo
done
