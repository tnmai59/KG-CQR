import os
import copy
import numpy as np
import pickle
import json
import requests
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import multiprocessing
from multiprocessing import Pool
from tqdm.notebook import tqdm
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
from transformers.utils import logging
logging.set_verbosity_error()
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, AutoTokenizer
import torch
from config import *
import pickle
import numpy as np
from embeddings import CustomAPIEmbeddings

# Environment setup
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# LLM API setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key="test",
    openai_api_base=INFERENCE_SERVER_URL,
    temperature=0,
    streaming=False
)

# Embedding API
embeddings = CustomAPIEmbeddings(api_url=EMBEDDING_API_URL, show_progress=False)


with open(TRIPLET_MAP_PATH,'rb') as f:
    dct_mapping_triplet = pickle.load(f)

with open(TRIPLET_EMB_PATH,'rb') as f:
    lst_embedding = pickle.load(f)

def faiss_cosine(query_vector, k=10):
    query_vector = query_vector.astype('float32')
    distances, indices = index.search(query_vector, k)
    return indices.flatten()

def query_triplet_topk(query, k=10):
    query_emb = np.array(embeddings.embed_query(query)).reshape(1,-1)
    topk_indices_sorted = faiss_cosine(query_emb).tolist()
    return [dct_mapping_triplet[x] for x in topk_indices_sorted]

lst_embedding = np.array(lst_embedding)
df_test = pd.read_json(TEST_DATA_PATH, lines=True)
test_data = df_test['question'].tolist()
df_test['documents'] = df_test['documents'].map(lambda x : eval(x))
faiss_embeddings = lst_embedding.astype('float32')
d = faiss_embeddings.shape[1]  
index_cpu = faiss.IndexFlatL2(d)
res = faiss.StandardGpuResources()  
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
index.add(faiss_embeddings)
lst_triplet_top_k_cos = []
for i in tqdm(test_data):
    lst_triplet_top_k_cos.append(query_triplet_topk(i))
map_triplet = {}
for i,j in zip(lst_triplet_top_k_cos, test_data):
    map_triplet[j] = i

import pickle

with open("cqr_res_ragbench.pkl","rb") as f:
    cqr_res = pickle.load(f)

with open("../data/processed_data/passages.txt","r") as f:
    lst_chunks = f.read().split("<endofpassage>")[:-1]
mapping_chunks = {j:i for i,j in enumerate(list(set(lst_chunks)))}
lst_chunks = list(set(lst_chunks))
lst_docs = lst_chunks
lst_label = []
for idx, row in tqdm(df_test.iterrows()):
	label = {mapping_chunks[x] for x in row['documents']}
	lst_label.append(label)

def recall_at_k(relevant_docs, retrieved_docs, k=25):
    k = min(k, len(retrieved_docs))
    retrieved_relevant_docs = set(retrieved_docs[:k]) & relevant_docs
    if len(relevant_docs) == 0:
        return 0
    recall = len(retrieved_relevant_docs) / len(relevant_docs)
    return recall

def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# BM25
corpus = [doc.split(" ") for doc in lst_docs]
bm25 = BM25Okapi(corpus)
def bm25_qcr(query):
    tokenized_query = query.split(" ")
    lst_retrieval = bm25.get_top_n(tokenized_query, lst_docs, n=25)
    return [mapping_chunks[x] for x in lst_retrieval]

# Multiprocessing for BM25 retrieval
n_jobs = 128
with multiprocessing.Pool(n_jobs) as pool:
    bm25_cqr = list(pool.imap(bm25_qcr, cqr_res))
base = [x.split(" with some extra data: ")[0] for x in cqr_res]
with multiprocessing.Pool(n_jobs) as pool:
    bm25_base = list(pool.imap(bm25_qcr, base))

# Evaluation output
for k in [5, 10, 15, 20, 25]:
    print(f"MAP@{k} : {mapk(lst_label, bm25_base, k)}")
    avg_recall_k = [recall_at_k(i, j, k) for i, j in zip(lst_label, bm25_base)]
    print(f"Average recall@{k} : {sum(avg_recall_k) / len(avg_recall_k)}")
print("*" * 100)
print("With contextual:")
for k in [5, 10, 15, 20, 25]:
    print(f"MAP@{k} : {mapk(lst_label, bm25_cqr, k)}")
    avg_recall_k = [recall_at_k(i, j, k) for i, j in zip(lst_label, bm25_cqr)]
    print(f"Average recall@{k} : {sum(avg_recall_k) / len(avg_recall_k)}") 

# BGE
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-large-en-v1.5',
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

embeddings_1 = model.encode(lst_docs, 
                            batch_size=64, 
                            max_length=512, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']

question = [x[0] for x in cqr_res]
context_cqr = [x[1] if len(x) ==2 else x[0] for x in cqr_res]
embeddings_q = model.encode(question, max_length=512, batch_size=n_jobs)['dense_vecs']
embeddings_3 = model.encode(context_cqr, max_length=512, batch_size=n_jobs)['dense_vecs']

import numpy as np
k = 25
chunk_size = 32  # Number of rows per chunk for the smaller matrix
def compute_topk_indices(large_matrix, small_chunk, k):
    similarity = small_chunk @ large_matrix.T  # Compute similarity
    top_k_indices = np.argsort(similarity, axis=1)[:, -k:][:, ::-1]  # Top-k indices in descending order
    return top_k_indices

def task(t):
    top_k_indices = compute_topk_indices(embeddings_1, t, 25)
    return top_k_indices
lst_task = []

for i in tqdm(range(0, embeddings_q.shape[0], chunk_size)):
	small_chunk = embeddings_q[i:i+chunk_size]
	lst_task.append(small_chunk)

with Pool(n_jobs) as pool:
	top_k_indices_list = list(tqdm(pool.imap(task, lst_task), total=len(lst_task)))

final_top_k_indices = np.vstack(top_k_indices_list)
bge = []
t = final_top_k_indices.tolist()
print("Without contextual")
for k in [5,10,15,20,25]:
	print(f"MAP@{k} : {mapk(lst_label, t, k)}")
	avg_recall_k = []
	for i, j in tqdm(zip(lst_label, t)):
		recall_value = recall_at_k(i, j, k)
		avg_recall_k.append(recall_value)
	print(f"Average recall@{k} : ", sum(avg_recall_k)/len(avg_recall_k))
	bge.append(sum(avg_recall_k)/len(avg_recall_k))
print("*"*100)

lst_bge = {"1.0": bge}

for alpha in [0.7, 0.5, 0.3, 0.0]:
	beta = 1 - alpha
	embeddings_2 = (alpha * embeddings_q + beta * embeddings_3)
	lst_task = []
	
	for i in tqdm(range(0, embeddings_2.shape[0], chunk_size)):
		small_chunk = embeddings_2[i:i+chunk_size]
		lst_task.append(small_chunk)
	
	with Pool(128) as pool:
		top_k_indices_list = list(tqdm(pool.imap(task, lst_task), total=len(lst_task)))
	
	final_top_k_indices = np.vstack(top_k_indices_list)
	bge = []
	t = final_top_k_indices.tolist()
	print(f"With contextual alpha {alpha}")
	for k in [5,10,15,20,25]:
		print(f"MAP@{k} : {mapk(lst_label, t, k)}")
		avg_recall_k = []
		for i, j in tqdm(zip(lst_label, t)):
			recall_value = recall_at_k(i, j, k)
			avg_recall_k.append(recall_value)
		print(f"Average recall@{k} : ", sum(avg_recall_k)/len(avg_recall_k))
		bge.append(sum(avg_recall_k)/len(avg_recall_k))
	lst_bge[str(alpha)] = bge
	print("*"*100)
print(lst_bge)

# DPR
# Load question encoder and tokenizer
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Load context encoder and tokenizer
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

passages = lst_docs

def encode_passages(passages, encoder, tokenizer):
    encoded = tokenizer(passages, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = encoder(**encoded).pooler_output
    return embeddings.numpy()

# Encode documents
dpr_doc_embeddings = encode_passages(lst_docs, context_encoder, context_tokenizer)

# Encode questions
question = [x[0] for x in cqr_res]
context_cqr = [x[1] if len(x) == 2 else x[0] for x in cqr_res]
dpr_q_embeddings = encode_passages(question, question_encoder, question_tokenizer)
dpr_cqr_embeddings = encode_passages(context_cqr, question_encoder, question_tokenizer)

def compute_dpr_topk_indices(query_embeddings, doc_embeddings, k=25, threshold=0.5):
    # Compute cosine similarity
    similarity = cosine_similarity(query_embeddings, doc_embeddings)
    # Apply threshold
    similarity[similarity < threshold] = 0
    # Get top-k indices
    top_k_indices = np.argsort(similarity, axis=1)[:, -k:][:, ::-1]
    return top_k_indices

# Process without contextual information
dpr_base_indices = compute_dpr_topk_indices(dpr_q_embeddings, dpr_doc_embeddings)
dpr_base_results = dpr_base_indices.tolist()

print("DPR Without contextual")
for k in [5, 10, 15, 20, 25]:
    print(f"MAP@{k} : {mapk(lst_label, dpr_base_results, k)}")
    avg_recall_k = [recall_at_k(i, j, k) for i, j in zip(lst_label, dpr_base_results)]
    print(f"Average recall@{k} : {sum(avg_recall_k) / len(avg_recall_k)}")

print("*" * 100)

# Process with contextual information using different alpha values
lst_dpr = {"1.0": []}  # Store results for different alpha values

for alpha in [0.7, 0.5, 0.3, 0.0]:
    beta = 1 - alpha
    # Combine embeddings with alpha-beta weighting
    combined_embeddings = (alpha * dpr_q_embeddings + beta * dpr_cqr_embeddings)
    
    # Compute top-k indices with threshold
    dpr_indices = compute_dpr_topk_indices(combined_embeddings, dpr_doc_embeddings)
    dpr_results = dpr_indices.tolist()
    
    print(f"DPR With contextual alpha {alpha}")
    for k in [5, 10, 15, 20, 25]:
        print(f"MAP@{k} : {mapk(lst_label, dpr_results, k)}")
        avg_recall_k = [recall_at_k(i, j, k) for i, j in zip(lst_label, dpr_results)]
        print(f"Average recall@{k} : {sum(avg_recall_k) / len(avg_recall_k)}")
        lst_dpr[str(alpha)].append(sum(avg_recall_k) / len(avg_recall_k))










