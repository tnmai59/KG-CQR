import os

# API Settings
INFERENCE_SERVER_URL = "http://127.0.0.1:9012/v1"
EMBEDDING_API_URL = "http://0.0.0.0:8000/get_emb"


# Model Settings
MODEL_NAME = "Meta-Llama-3-70B-Instruct"
TEMPERATURE = 0

# File paths
TRIPLET_MAP_PATH = "triplet_map.pkl"
TRIPLET_EMB_PATH = "triplet_emb.pkl"
TEST_DATA_PATH = "../data/raw_data/final_data.jsonl"
OUTPUT_PATH = "cqr_res.pkl"
EMBEDDING_PATH = "../HUB_Embeddings/bge-m3"
CORPUS_PATH = "../data/raw_data/corpus.pkl"

# Processing settings
NUM_PROCESSES = 10
TOP_K = 10 
BEAM_WIDTH = 20
MAX_PATH_LENGTH = 2
MAX_NEW_TRIPLETS = 20
NUM_PROCESSES = 10
TOP_K = 25