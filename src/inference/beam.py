import pickle
import json
import requests
from collections import defaultdict, namedtuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import CustomAPIEmbeddings
from config import *

# —————————————————————————————————————————————
#  Data structures & utilities
# —————————————————————————————————————————————

Triplet = namedtuple("Triplet", ["head", "relation", "tail", "ttr"])

def format_T(x):
	"""
	Convert your loaded format into the {'m','r','n'} dict format.
	"""
	return [{
		"m": {"id": i['n']['id']},
		"r": {"id": i['r'][1], "summary": i['r.summary']},
		"n": {"id": i['m']['id']}
	} for i in x]

def normalize_triplet(t):
	"""
	If you ever added '_rev' relations, normalize them back.
	"""
	m, r, n = t['m']['id'], t['r']['id'], t['n']['id']
	summary = t['r']['summary']
	if r.endswith("_rev"):
		r = r[:-4]
		m, n = n, m
	return {"m": {"id": m}, "r": {"id": r, "summary": summary}, "n": {"id": n}}

def build_undirected_graph(triplets):
	"""
	Build adjacency list: node_id -> list of edges.
	Each edge is {'m':{'id'}, 'r':{'id','summary'}, 'n':{'id'}}.
	Only forward edges by default; add reverse if needed.
	"""
	graph = defaultdict(list)
	for t in triplets:
		graph[t.head].append({
			"m": {"id": t.head},
			"r": {"id": t.relation, "summary": t.ttr},
			"n": {"id": t.tail}
		})
		# Uncomment to add reverse edges:
		# graph[t.tail].append({
		#     "m": {"id": t.tail},
		#     "r": {"id": t.relation + "_rev", "summary": t.ttr},
		#     "n": {"id": t.head}
		# })
	return graph

# —————————————————————————————————————————————
#  Beam search functions
# —————————————————————————————————————————————

def beam_search_paths(KG, start_entities, end_entities,
					  summary_embeddings, question_emb,
					  beam_width=10, max_depth=5):
	"""
	Beam search over KG.
	  - KG: adjacency list from build_undirected_graph
	  - start_entities: iterable of node IDs
	  - end_entities: set of node IDs to consider as 'completed'
	  - summary_embeddings: dict[str] -> np.ndarray
	  - question_emb: np.ndarray or None
	  - beam_width: how many paths to keep each round
	  - max_depth: max hops

	Returns list of (path, score), sorted descending.
	"""
	# beam entries: (path_list_of_edges, last_node, score, visited_set)
	beams = [([], node, 0.0, {node}) for node in start_entities]
	completed = []

	for depth in range(1, max_depth + 1):
		candidates = []
		for path, last_node, score, visited in beams:
			for edge in KG.get(last_node, []):
				nbr = edge["n"]["id"]
				if nbr in visited:
					continue
				emb = summary_embeddings[edge["r"]["summary"]]
				sim = 0.0
				if question_emb is not None:
					sim = cosine_similarity(
						np.array(emb).reshape(1, -1),
						np.array(question_emb).reshape(1, -1)
					)[0, 0]
				new_score = score + sim
				candidates.append((path + [edge], nbr, new_score, visited | {nbr}))

		# pick top beam_width
		candidates.sort(key=lambda x: x[2], reverse=True)
		if len(candidates) > 100:
			print(len(candidates))
		beams = candidates[:beam_width]

		# split completed vs. keep-going
		next_beams = []
		for path, node, sc, vis in beams:
			if node in end_entities:
				completed.append((path, sc))
			else:
				next_beams.append((path, node, sc, vis))
		beams = next_beams

		if not beams:
			break

	# return sorted completed paths
	return sorted(completed, key=lambda x: x[1], reverse=True)

def relevance_guided_path_addition_beam(KG, T, question, model,
										summary_embeddings,
										K=100, beam_width=20, max_path_length=2):
	"""
	Extend subgraph T by up to K new triplets found via beam search.
	"""
	# entities in T
	E_T = {t["m"]["id"] for t in T} | {t["n"]["id"] for t in T}

	# question embedding
	question_emb = None
	if question:
		question_emb = model.embed_query([question])[0]

	# keys of original T
	T_keys = {(t["m"]["id"], t["r"]["id"], t["n"]["id"]) for t in T}

	# find paths
	completed = beam_search_paths(
		KG,
		start_entities=E_T,
		end_entities=E_T,
		summary_embeddings=summary_embeddings,
		question_emb=question_emb,
		beam_width=beam_width,
		max_depth=max_path_length
	)

	if not completed:
		return T

	# flatten and dedupe new triplets
	selected = []
	sel_keys = set()
	for path, _ in completed:
		for triplet in path:
			norm = normalize_triplet(triplet)
			key = (norm["m"]["id"], norm["r"]["id"], norm["n"]["id"])
			if key not in T_keys and key not in sel_keys:
				selected.append(norm)
				sel_keys.add(key)
				if len(selected) >= K:
					break
		if len(selected) >= K:
			break

	return T + selected
# —————————————————————————————————————————————
#  Main execution
# —————————————————————————————————————————————

if __name__ == "__main__":
	# 1) Load and format KG
	with open(TRIPLET_MAP_PATH, "rb") as f:
		raw_triplets = pickle.load(f)

	KG_list = [
		Triplet(rec["r"][0]["id"], rec["r"][1], rec["r"][2]["id"], rec["r.summary"])
		for rec in raw_triplets
	]
	KG = build_undirected_graph(KG_list)

	# 2) Precompute all summary embeddings
	all_summaries = {edge["r"]["summary"] for edges in KG.values() for edge in edges}
	model = CustomAPIEmbeddings()
	summary_embeddings = dict(
		zip(
			list(all_summaries),
			model.embed_documents(list(all_summaries))
		)
	)
	# 3) Load questions and existing subgraphs
	with open("subgraph.pkl", "rb") as f:
		subgraph = pickle.load(f)

	with open(TEST_DATA_PATH,'r') as f:
		question_list = [x['query'] for x in json.load(f)]

	# 4) Iterate and extend
	cnt = 0
	for raw_T, question in tqdm(zip(subgraph, question_list), total=len(question_list)):
		T = format_T(raw_T)
		H = relevance_guided_path_addition_beam(
			KG, T, question, model,
			summary_embeddings,
			K=20, beam_width=20, max_path_length=2
		)
		if len(H) != len(T):
			cnt += 1
			# print("\nExtended subgraph H:")
			# for trip in H:
			# 	if trip in T:
			# 		continue
			# 	m, r, n = trip["m"]["id"], trip["r"]["id"], trip["n"]["id"]
			# 	print(f"  {m} -[{r}]-> {n}")
		# 	break
	print(f"\nTotal subgraphs extended: {cnt}")