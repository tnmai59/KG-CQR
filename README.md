# KG-CQR: Leveraging Structured Relation Representations in Knowledge Graphs for Contextual Query Retrieval

This project leverages a knowledge graph to improve the retrieval phase of contextual questions.

## Paper

https://aclanthology.org/2025.emnlp-main.824.pdf

## Environment Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- At least 16GB RAM
- At least 50GB free disk space

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/KG-Contextual-Question-Retrieval.git
cd KG-Contextual-Question-Retrieval
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```env
INFERENCE_SERVER_URL=http://127.0.0.1:9012/v1
EMBEDDING_API_URL=http://0.0.0.0:8000/get_emb
MODEL_NAME=Meta-Llama-3-70B-Instruct
```

## Project Structure

- `src/`: Source code for the project
  - `inference/`: Core inference pipeline and utilities
  - `embeddings/`: Embedding generation and management
  - `knowledge_graph/`: Knowledge graph construction and management
  - `data/`: Data processing and utilities
  - `tests/`: Test files and experiments
- `requirements.txt`: Project dependencies
- `setup.py`: Project setup configuration

## Usage

### 1. Start the Embedding Service

```bash
python3 ./src/embeddings/embeds.py \
    --model_path "/path/to/bge-large-en-v1.5" \
    --device "cuda" \
    --device-id 0 \
    --port 8000
```

### 2. Create Knowledge Graph

```bash
python3 ./src/knowledge_graph/knowledge_graph.py \
    --model_name "Meta-Llama-3-70B-Instruct" \
    --inference_server_url "http://127.0.0.1:9012/v1/" \
    --openai_api_key "your_api_key" \
    --embedding_api_url "http://127.0.0.1:8000/get_emb" \
    --input_file "./src/data/processed_data/ragbench.pkl" \
    --output_file "./src/data/processed_data/ragbench_graph.pkl" \
    --use_async \
    --num_proc 4
```

### 3. Run the Pipeline

```bash
python3 ./src/inference/pipeline.py
```

The pipeline will:
1. Load pre-computed triplet embeddings and mappings
2. Process test questions to find relevant triplets
3. Use beam search to find additional relevant paths
4. Use LLM to check relevance and generate contextual summaries
5. Save the results

### 4. Run Latency Analysis

```bash
python3 ./src/inference/latency.py \
    --beam-width 20 \
    --max-path-length 2 \
    --k 20
```

## Configuration

Key configuration parameters in `src/inference/config.py`:

- `BEAM_WIDTH`: Number of paths to maintain during beam search (default: 20)
- `MAX_PATH_LENGTH`: Maximum path length to search (default: 2)
- `MAX_NEW_TRIPLETS`: Maximum number of new triplets to add (default: 20)
- `NUM_PROCESSES`: Number of parallel processes for question processing
- `TOP_K`: Number of top-k triplets to retrieve

## Features

- Knowledge graph-based contextual question retrieval
- Beam search for finding relevant paths
- Parallel processing for improved performance
- Integration with Meta-Llama-3-70B-Instruct for relevance checking
- FAISS for efficient similarity search
- Custom API embeddings for text embeddings

## Citation

If you are interested or inspired by this work, you can cite us by:

```
@inproceedings{bui-etal-2025-kg,
    title = "{KG}-{CQR}: Leveraging Structured Relation Representations in Knowledge Graphs for Contextual Query Retrieval",
    author = "Bui, Chi Minh  and
      Thieu, Ngoc Mai  and
      Van Nguyen, Vinh  and
      Jung, Jason J.  and
      Bui, Khac-Hoai Nam",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.824/",
    doi = "10.18653/v1/2025.emnlp-main.824",
    pages = "16281--16298",
    ISBN = "979-8-89176-332-6"
}
```

## Authors

- buichiminh.cntt@gmail.com
- hoainam.bk2012@gmail.com
- thieungocmai.watt@gmail.com


