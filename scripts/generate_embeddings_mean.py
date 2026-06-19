import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import argparse

# ------------------------
# Config
# ------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--benchmark",
    type=str,
    choices=["RQ", "SRAQ", ],
    default="RQ",
    help="Which benchmark CSV to load (without .csv extension).",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen3-4B",
    help="HF model name (e.g. Qwen/Qwen3-4B).",
)

parser.add_argument(
    "--col",
    type=str,
    default="question_with_context",
    help="Column name in the CSV to encode.",
)

args = parser.parse_args()

benchmark = args.benchmark          
model_name = args.model_name        # e.g. "Qwen/Qwen3-4B"
model = model_name.split("/")[-1]   # e.g. "Qwen3-4B"
col = args.col                      # e.g. "question_with_context"

output_path = f"embeddings_mean/{benchmark}_{col}_{model}.pt"
os.makedirs("embeddings_mean", exist_ok=True)

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
# Extract last-token embeddings from every layer
# ------------------------
embeddings = []  # each element: [num_layers+1, hidden_dim] for one example

with torch.no_grad():
    for text in tqdm(texts):
        # batch size = 1
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states           # tuple(len = num_layers+1)
        last_token_embeddings_per_layer = [
            h[0, :, :].mean(dim=0, keepdim=False)   # use -1 here
            for h in hidden_states
        ]
        embeddings.append(torch.stack(last_token_embeddings_per_layer, dim=0))  # [num_layers+1, hidden_dim]

# ------------------------
embeddings = torch.stack(embeddings, dim=0)  # [num_examples, num_layers+1, hidden_dim]
torch.save({
    "embeddings": embeddings,
    'model_name': model_name,
    'benchmark': benchmark,
    'column': col
}, output_path)

print(f"Saved embeddings to: {output_path}")
