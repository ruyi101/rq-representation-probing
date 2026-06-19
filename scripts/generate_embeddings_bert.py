import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import argparse

# Enable TF32 for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')

# ------------------------
# Config
# ------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--benchmark",
    type=str,
    choices=["RQ", "SRAQ",],
    default="RQ",
    help="Which benchmark CSV to load (without .csv extension).",
)

parser.add_argument(
    "--col",
    type=str,
    default="question_with_context",
    help="Column name in the CSV to encode.",
)

args = parser.parse_args()

benchmark = args.benchmark          # e.g. "RQ"
model_name = "answerdotai/ModernBERT-large"  # Fixed model
model = "ModernBERT-large"
col = args.col                      # e.g. "question_with_context"

output_path = f"embeddings_bert/{benchmark}_{col}_{model}.pt"
os.makedirs("embeddings_bert", exist_ok=True)

# ------------------------
# Load data
# ------------------------
df = pd.read_csv(f"{benchmark}.csv")
texts = df[col].astype(str).tolist()

# ------------------------
# Load tokenizer & model
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)


model = AutoModel.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    output_hidden_states=True,  # make sure we get all layers
)
model.eval()

# ------------------------
# Extract CLS token embeddings from every layer
# ------------------------
embeddings = []  # each element: [num_layers+1, hidden_dim] for one example

with torch.no_grad():
    for text in tqdm(texts):
        # batch size = 1
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states           # tuple(len = num_layers+1)
        cls_token_embeddings_per_layer = [
            h[0, 0, :]   # use 0 for CLS token
            for h in hidden_states
        ]
        embeddings.append(torch.stack(cls_token_embeddings_per_layer, dim=0))  # [num_layers+1, hidden_dim]

# ------------------------
embeddings = torch.stack(embeddings, dim=0)  # [num_examples, num_layers+1, hidden_dim]
torch.save({
    "embeddings": embeddings,
    'model_name': model_name,
    'benchmark': benchmark,
    'column': col
}, output_path)

print(f"Saved embeddings to: {output_path}")
