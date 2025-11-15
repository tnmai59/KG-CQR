from transformers import AutoTokenizer
import numpy as np
import faiss
import logging
import pickle
from config import *
from embeddings import CustomAPIEmbeddings

ebd_tok = AutoTokenizer.from_pretrained(EMBEDDING_PATH)
embeddings = CustomAPIEmbeddings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def max_len(query, thres=512):
	t = ebd_tok.encode(query)
	if len(t) > thres:
		t = t[:(thres-12)]
		query = ebd_tok.decode(t)
	return query

def faiss_cosine(index, query_vector, k=10):
	query_vector = query_vector.astype('float32')
	distances, indices = index.search(query_vector, k)
	return indices.flatten()

def query_triplet_topk(query, k=10):
	query = max_len(query)
	query_emb = np.array(embeddings.embed_query(query)).reshape(1,-1)
	topk_indices_sorted = faiss_cosine(query_emb).tolist()
	return topk_indices_sorted

def format_relations(relations):
	result = []
	for rel in relations:
		formatted_relation = f"{rel['n']['id']} - {rel['r'][1]} -> {rel['m']['id']}"
		result.append(formatted_relation)
	return result

def format_claim(relations):
	return "\n\n".join(f"{idx+1}. {rel['r']['summary']}" for idx, rel in enumerate(relations))

def format_triplet(relations):
	return "\n\n".join(f"{idx+1}. ({rel['r'][0]['id']}, {rel['r'][1]}, {rel['r'][2]['id']})" for idx, rel in enumerate(relations))

def format_docs(docs):
	return "\n\n".join(f"{doc}" for doc in docs)

def max_length_context(context,threshold=512):
	res = []
	for i in context:
		if len(i.split(" ")) > threshold:
			tmp = " ".join(x for x in i.split(" ")[:threshold])
			res.append(tmp)
		else:
			res.append(i)
	return res

def initialize_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Initialize FAISS index for similarity search.
    
    Args:
        embeddings: Array of embeddings to index
        
    Returns:
        Initialized FAISS index
    """
    try:
        faiss_embeddings = embeddings.astype('float32')
        d = faiss_embeddings.shape[1]
        index_cpu = faiss.IndexFlatL2(d)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        index.add(faiss_embeddings)
        return index
    except Exception as e:
        logger.error(f"Error initializing FAISS index: {str(e)}")
        raise

def load_triplet_data(triplet_map_path, triplet_emb_path) -> tuple:
    """Load triplet mapping and embeddings from pickle files.
    
    Returns:
        tuple: (dct_mapping_triplet, lst_embedding)
            - dct_mapping_triplet: Dictionary mapping indices to triplets
            - lst_embedding: List of triplet embeddings
    """
    try:
        with open(triplet_map_path, 'rb') as f:
            dct_mapping_triplet = pickle.load(f)
        
        with open(triplet_emb_path, 'rb') as f:
            lst_embedding = pickle.load(f)
            
        return dct_mapping_triplet, np.array(lst_embedding)
    except Exception as e:
        logger.error(f"Error loading triplet data: {str(e)}")
        raise

def load_corpus(corpus_path):
	with open(corpus_path,"rb") as f:
		lst_chunks = pickle.load(corpus_path)
	mapping_chunks = {j:i for i,j in enumerate(list(set(lst_chunks)))}
	return mapping_chunks