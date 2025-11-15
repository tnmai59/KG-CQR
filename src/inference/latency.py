import pickle
import json
import requests
import argparse
from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from config import *
from beam import *
from embeddings import CustomAPIEmbeddings
# —————————————————————————————————————————————
#  Data structures & utilities
# —————————————————————————————————————————————

Triplet = namedtuple("Triplet", ["head", "relation", "tail", "ttr"])


# —————————————————————————————————————————————
#  Main execution
# —————————————————————————————————————————————

def parse_args():
	parser = argparse.ArgumentParser(description='Knowledge Graph Path Search')
	parser.add_argument('--beam-width', type=int, default=20,
					  help='Width of the beam search (default: 20)')
	parser.add_argument('--max-path-length', type=int, default=2,
					  help='Maximum path length to search (default: 2)')
	parser.add_argument('--k', type=int, default=20,
					  help='Maximum number of new triplets to add (default: 20)')
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	model = CustomAPIEmbeddings()
	
	# 1) Load and format KG
	with open(TRIPLET_MAP_PATH, "rb") as f:
		raw_triplets = pickle.load(f)

	KG_list = [
		Triplet(rec["r"][0]["id"], rec["r"][1], rec["r"][2]["id"], rec["r.summary"])
		for rec in raw_triplets
	]
	KG = build_undirected_graph(KG_list)

	# 2) Load summary embeddings
	with open(TRIPLET_EMB_PATH,'rb') as f:
		summary_embeddings = pickle.load(f)

	# 3) Load questions and existing subgraphs
	with open("subgraph.pkl", "rb") as f:
		subgraph = pickle.load(f)
	with open(TEST_DATA_PATH,'r') as f:
		question_list = [x['query'] for x in json.load(f)]

	# 4) Iterate and extend
	cnt = 0
	for raw_T, question in tqdm(zip(subgraph, question_list)):
		T = format_T(raw_T)
		H = relevance_guided_path_addition_beam(
			KG, T, question, model,
			summary_embeddings,
			K=args.k,
			beam_width=args.beam_width,
			max_path_length=args.max_path_length
		)