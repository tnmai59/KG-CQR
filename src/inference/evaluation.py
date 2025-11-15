import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets
from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import faiss

def load_data():
    lst_ds = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']
    lst_test = []
    for i in tqdm(lst_ds):
        tmp_ds = load_dataset("rungalileo/ragbench", i, split="test")
        lst_test.append(tmp_ds)
    lst_chunks = []
    for i in lst_test:
        for j in i:
            lst_chunks.extend(j['documents'])
    lst_chunks = list(set(lst_chunks))
    mapping_chunks = {j:i for i,j in enumerate(lst_chunks)}
    final_ds = concatenate_datasets(lst_test)
    return lst_chunks, mapping_chunks, final_ds

def recall_at_k(relevant_docs, retrieved_docs, k=25):
    k = min(k, len(retrieved_docs))
    retrieved_relevant_docs = set(retrieved_docs[:k]) & relevant_docs
    if len(relevant_docs) == 0:
        return 0
    recall = len(retrieved_relevant_docs) / len(relevant_docs)
    return recall

def evaluate_bm25(lst_chunks, final_ds, mapping_chunks):
    tokenized_corpus = [doc.split(" ") for doc in lst_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    lst_recall = {}
    for k in [5, 10, 15, 20, 25]:
        avg_recall_k = []
        for idx, i in tqdm(enumerate(final_ds)):
            query = i['question']
            tokenized_query = query.split(" ")
            topk = i['documents']
            label = set([mapping_chunks[x] for x in topk])
            lst_retrieval = bm25.get_top_n(tokenized_query, lst_chunks, n=k)
            pred = [mapping_chunks[x] for x in lst_retrieval]
            recall_value = recall_at_k(label, pred, k)
            avg_recall_k.append(recall_value)
        print(f"Average recall@{k} : ", sum(avg_recall_k)/len(avg_recall_k))
        lst_recall[k] = sum(avg_recall_k)/len(avg_recall_k)
    return lst_recall

def evaluate_dpr(lst_chunks, final_ds, mapping_chunks):
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    def encode_passages(passages, encoder, tokenizer):
        encoded_passages = []
        for passage in tqdm(passages):
            inputs = tokenizer(passage, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            embeddings = encoder(**inputs).pooler_output
            encoded_passages.append(embeddings.cpu().detach().numpy())
        return np.vstack(encoded_passages)
    
    encoded_passages = encode_passages(lst_chunks, context_encoder, context_tokenizer)
    index = faiss.IndexFlatIP(encoded_passages.shape[1])
    index.add(encoded_passages)
    
    lst_recall = {}
    for k in [5, 10, 15, 20, 25]:
        avg_recall_k = []
        for idx, i in tqdm(enumerate(final_ds)):
            question = i['question']
            topk = i['documents']
            label = set([mapping_chunks[x] for x in topk])
            inputs = question_tokenizer(question, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            question_embedding = question_encoder(**inputs).pooler_output.cpu().detach().numpy()
            distances, indices = index.search(question_embedding, k)
            pred = indices[0].tolist()
            recall_value = recall_at_k(label, pred, k)
            avg_recall_k.append(recall_value)
        print(f"Average recall@{k} : ", sum(avg_recall_k)/len(avg_recall_k))
        lst_recall[k] = sum(avg_recall_k)/len(avg_recall_k)
    return lst_recall

def evaluate_dense_model(lst_chunks, final_ds, mapping_chunks, model_name):
    model = SentenceTransformer(model_name)
    q_emb = model.encode(final_ds['question'], show_progress_bar=True)
    d_emb = model.encode(lst_chunks, show_progress_bar=True)
    
    def compute_topk_indices(large_matrix, small_chunk, k):
        similarity = small_chunk @ large_matrix.T
        top_k_indices = np.argsort(similarity, axis=1)[:, -k:][:, ::-1]
        return top_k_indices
    
    chunk_size = 32
    top_k_indices_list = []
    for i in tqdm(range(0, q_emb.shape[0], chunk_size)):
        small_chunk = q_emb[i:i+chunk_size]
        top_k_indices = compute_topk_indices(d_emb, small_chunk, 25)
        top_k_indices_list.append(top_k_indices)
    
    final_top_k_indices = np.vstack(top_k_indices_list)
    
    lst_recall = {}
    for k in [5, 10, 15, 20, 25]:
        avg_recall_k = []
        for i, j in tqdm(zip(final_ds['documents'], final_top_k_indices)):
            label = set([mapping_chunks[x] for x in i])
            recall_value = recall_at_k(label, j, k)
            avg_recall_k.append(recall_value)
        print(f"Average recall@{k} : ", sum(avg_recall_k)/len(avg_recall_k))
        lst_recall[k] = sum(avg_recall_k)/len(avg_recall_k)
    return lst_recall

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval models")
    parser.add_argument("--model", type=str, choices=["bm25", "dpr", "dense"], required=True, help="Model to evaluate")
    parser.add_argument("--dense_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Dense model name")
    args = parser.parse_args()
    
    lst_chunks, mapping_chunks, final_ds = load_data()
    
    if args.model == "bm25":
        evaluate_bm25(lst_chunks, final_ds, mapping_chunks)
    elif args.model == "dpr":
        evaluate_dpr(lst_chunks, final_ds, mapping_chunks)
    elif args.model == "dense":
        evaluate_dense_model(lst_chunks, final_ds, mapping_chunks, args.dense_model_name)

if __name__ == "__main__":
    main()
