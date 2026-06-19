#!/usr/bin/env bash
set -euo pipefail

# ----------- basic config (edit these) -----------

EMB_DIR="../embeddings_synthetic_last"        # folder containing *.pt embeddings
DATASET_DIR="../"      # folder containing benchmark CSVs

BENCHMARK="Hong-1201"            # e.g. SRAQ, RQ, ...
COLUMN="question_with_context"          # e.g. paragraph, question, ...
MODEL="Llama-3.1-8B-Instruct"           # e.g. gemma2-9b, llama-3-8b, ...

LOSS_TYPE="logistic"        # logistic or hinge
LR=0.001
WEIGHT_DECAY=0.01
MAX_EPOCHS=500
PATIENCE=20
BATCH_SIZE=256

# ----------- optional CLI overrides ------------
# You can optionally override BENCHMARK/COLUMN/MODEL from the command line:
#   ./run_linear.sh SRAQ paragraph gemma2-9b
#
# If you don't pass them, the defaults above are used.

if [[ $# -ge 1 ]]; then
  BENCHMARK="$1"
fi

if [[ $# -ge 2 ]]; then
  COLUMN="$2"
fi

if [[ $# -ge 3 ]]; then
  MODEL="$3"
fi

echo "Running linear training with:"
echo "  EMB_DIR      = ${EMB_DIR}"
echo "  DATASET_DIR  = ${DATASET_DIR}"
echo "  BENCHMARK    = ${BENCHMARK}"
echo "  COLUMN       = ${COLUMN}"
echo "  MODEL        = ${MODEL}"
echo "  LOSS_TYPE    = ${LOSS_TYPE}"
echo "  LR           = ${LR}"
echo "  WEIGHT_DECAY = ${WEIGHT_DECAY}"
echo "  MAX_EPOCHS   = ${MAX_EPOCHS}"
echo "  PATIENCE     = ${PATIENCE}"
echo "  BATCH_SIZE   = ${BATCH_SIZE}"

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
