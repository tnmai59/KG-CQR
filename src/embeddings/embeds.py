import os
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Argument parsing
parser = argparse.ArgumentParser(description='Embedding Service')
parser.add_argument('--model_path', type=str, default='/raid/HUB_LLM/bge-large-en-v1.5', help='Path to the model')
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use for computation')
parser.add_argument('--device-id', type=int, default=0, help='Device ID to use for computation')
parser.add_argument('--port', type=int, default=8000, help='Port to run the FastAPI server on')
parser.add_argument('--workers', type=int, default=8, help='Set number of workers for faster')

args = parser.parse_args()

# Set CUDA_VISIBLE_DEVICES
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

import torch

# Use remapped device id
device = torch.device('cuda:0') if args.device == 'cuda' else torch.device('cpu')

print(f"RUNNING MODEL ON CUDA DEVICE {args.device_id}")

# Load model on the specified device
if args.device == 'cuda':
    encoder = SentenceTransformer(args.model_path).to(device)
else:
    encoder = SentenceTransformer(args.model_path)


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

def sort_dictionaries_by_cosine_similarity(dictionaries, target_vector):
    sorted_dictionaries = sorted(dictionaries, key=lambda d: cosine_similarity(d["rerank_emb"], target_vector), reverse=True)
    sorted_scores = [(d, cosine_similarity(d['rerank_emb'], target_vector)) for d in sorted_dictionaries]
    return sorted_scores


app = FastAPI()

class Query(BaseModel):
    query: str
    top_k: int

class Embedding_m(BaseModel):
    query: str

@app.post("/get_emb")
async def get_emb(item: Embedding_m):
    query = item.query
    out = encoder.encode(query).tolist()
    return {"embedding":out, "status":200}

if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path

    # Determine the current script's module path
    module_name = Path(sys.argv[0]).stem  # Gets the filename without extension
    app_import_path = f"{module_name}:app"

    # Use Uvicorn with reload or workers
    uvicorn.run(
        app_import_path,
        host="0.0.0.0",
        port=args.port,
        workers=args.workers
    )