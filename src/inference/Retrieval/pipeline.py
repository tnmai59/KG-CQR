import os
from tqdm.notebook import tqdm
import copy
import numpy as np
import pickle
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple
from multiprocessing import Pool
import logging
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI
import faiss
from embeddings import CustomAPIEmbeddings
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from config import *
tokenizer = AutoTokenizer.from_pretrained(INFERENCE_SERVER_URL)

llm = ChatOpenAI(
    model="Meta-Llama-3-70B-Instruct",
    openai_api_key="test",
    openai_api_base=INFERENCE_SERVER_URL,
    temperature=0,
    streaming= False
)
# Initialize embeddings
embeddings = CustomAPIEmbeddings(api_url=EMBEDDING_API_URL, show_progress=False)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_triplet_data() -> tuple:
    """Load triplet mapping and embeddings from pickle files.
    
    Returns:
        tuple: (dct_mapping_triplet, lst_embedding)
            - dct_mapping_triplet: Dictionary mapping indices to triplets
            - lst_embedding: List of triplet embeddings
    """
    try:
        with open(TRIPLET_MAP_PATH, 'rb') as f:
            dct_mapping_triplet = pickle.load(f)
        
        with open(TRIPLET_EMB_PATH, 'rb') as f:
            lst_embedding = pickle.load(f)
            
        return dct_mapping_triplet, np.array(lst_embedding)
    except Exception as e:
        logger.error(f"Error loading triplet data: {str(e)}")
        raise

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

# Load data and initialize index
dct_mapping_triplet, lst_embedding = load_triplet_data()
index = initialize_faiss_index(lst_embedding)

def faiss_cosine(query_vector: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Search for nearest neighbors using FAISS.
    
    Args:
        query_vector: Query vector to search for
        k: Number of nearest neighbors to return
        
    Returns:
        Array of indices of nearest neighbors
    """
    try:
        query_vector = query_vector.astype('float32')
        distances, indices = index.search(query_vector, k)
        return indices.flatten()
    except Exception as e:
        logger.error(f"Error in FAISS search: {str(e)}")
        raise

def query_triplet_topk(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Get top-k most relevant triplets for a query.
    
    Args:
        query: Query string
        k: Number of triplets to return
        
    Returns:
        List of relevant triplets with their metadata
    """
    try:
        query_emb = np.array(embeddings.embed_query(query)).reshape(1, -1)
        topk_indices_sorted = faiss_cosine(query_emb, k).tolist()
        return [dct_mapping_triplet[x] for x in topk_indices_sorted]
    except Exception as e:
        logger.error(f"Error querying triplets: {str(e)}")
        raise

def format_claim(relations: List[Dict[str, Any]]) -> str:
    """Format relations into a claim string.
    
    Args:
        relations: List of relations to format
        
    Returns:
        Formatted claim string with numbered entries
    """
    try:
        for rel in relations:
            rel['r.summary'] = rel['r.summary'].split("\n\n")[-1]
        return "\n\n".join(f"{idx+1}. {rel['r.summary']}" for idx, rel in enumerate(relations))
    except Exception as e:
        logger.error(f"Error formatting claims: {str(e)}")
        raise

def check_relations(question: str, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Check which relations are relevant to the question.
    
    Args:
        question: Question to check relevance against
        relations: List of relations to check
        
    Returns:
        List of relevant relations
    """
    try:
        result = []
        for rel in relations:
            check = check_grade(question, rel['r.summary'])
            if check.binary_score == "yes":
                result.append(rel)
        return result
    except Exception as e:
        logger.error(f"Error checking relations: {str(e)}")
        raise

def format_relations(relations):
    result = []
    for rel in relations:
        formatted_relation = f"{rel['n']['id']} - {rel['r'][1]} -> {rel['m']['id']}"
        result.append(formatted_relation)
    return result

import traceback

cnt_err = 0
def format_claim(relations):
    for rel in relations:
        rel['r.summary'] = rel['r.summary'].split("\n\n")[-1]
    # return "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs))
    return "\n\n".join(f"{idx+1}. {rel['r.summary']}" for idx, rel in enumerate(relations))

def format_triplet(relations):
    return "\n\n".join(f"{idx+1}. ({rel['r'][0]['id']}, {rel['r'][1]}, {rel['r'][2]['id']})" for idx, rel in enumerate(relations))


class contextual_output(BaseModel):
    """contextual summarization for the input question."""
    summary: str = Field(
        description="Concise summary contextual information of the input question"
    )

class contextual_triplets(BaseModel):
    """contextual generation of knowledge subgraph."""
    context: str = Field(
        description="generate contextual information based on list of triplets and their descriptions"
    )
    

def contextual_question_retrieval(claims):
    system_promt="You are a helpful assistant responsible for generating a comprehensive summary of the data provided below."
    " Given the list of claims that may relation with each other. Please write a Concise summary of claims that aim to provide a contextual information."
    " The output just generate a concise summary without any explaination."
    " Please note that if the provided claims are contradictory, please resolve the contradictions and provide a single, coherent summary (no need Here is part)"
    chat_template_contextual = tokenizer.apply_chat_template(
        [
            {"role":"system", "content":"{system}"},
            {"role":"user", "content":"\nInput Claims: {claims}\n"}
        ], tokenize=False, add_generation_prompt=True)
    
    prompt_summary_contextual = PromptTemplate(template=chat_template_contextual, input_variables=["system", "claims"])
    structured_summary_contextual = llm.with_structured_output(contextual_output)
    contextual_chain = prompt_summary_contextual | structured_summary_contextual 
    results = contextual_chain.invoke({"system": system_promt, "claims": claims})
    return results

def format_triplet_mixed(relations):
    for rel in relations:
        rel['r.summary'] = rel['r.summary'].split("\n\n")[-1]
    return "\n".join(f"({rel['n']['id']}, {rel['r'][1]}, {rel['m']['id']}): {rel['r.summary']}" for idx, rel in enumerate(relations))

def build_undirected_graph(triplets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build adjacency list representation of the knowledge graph.
    
    Args:
        triplets: List of triplets in the format {'m': {'id': str}, 'r': {'id': str, 'summary': str}, 'n': {'id': str}}
        
    Returns:
        Dictionary mapping node IDs to their outgoing edges
    """
    graph = defaultdict(list)
    for t in triplets:
        graph[t['m']['id']].append(t)
    return graph

def beam_search_paths(
    kg: Dict[str, List[Dict[str, Any]]],
    start_entities: Set[str],
    end_entities: Set[str],
    summary_embeddings: Dict[str, np.ndarray],
    question_emb: Optional[np.ndarray],
    beam_width: int = BEAM_WIDTH,
    max_depth: int = MAX_PATH_LENGTH
) -> List[Tuple[List[Dict[str, Any]], float]]:
    """Perform beam search to find relevant paths in the knowledge graph.
    
    Args:
        kg: Knowledge graph as adjacency list
        start_entities: Set of starting node IDs
        end_entities: Set of target node IDs
        summary_embeddings: Dictionary mapping relation summaries to their embeddings
        question_emb: Question embedding for relevance scoring
        beam_width: Number of paths to maintain at each step
        max_depth: Maximum path length to search
        
    Returns:
        List of (path, score) tuples, sorted by score descending
    """
    beams = [([], node, 0.0, {node}) for node in start_entities]
    completed = []

    for depth in range(1, max_depth + 1):
        candidates = []
        for path, last_node, score, visited in beams:
            for edge in kg.get(last_node, []):
                nbr = edge["n"]["id"]
                if nbr in visited:
                    continue
                    
                # Calculate relevance score
                emb = summary_embeddings[edge["r"]["summary"]]
                sim = 0.0
                if question_emb is not None:
                    sim = cosine_similarity(
                        np.array(emb).reshape(1, -1),
                        np.array(question_emb).reshape(1, -1)
                    )[0, 0]
                new_score = score + sim
                candidates.append((path + [edge], nbr, new_score, visited | {nbr}))

        # Select top beam_width candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]

        # Split completed vs continuing paths
        next_beams = []
        for path, node, sc, vis in beams:
            if node in end_entities:
                completed.append((path, sc))
            else:
                next_beams.append((path, node, sc, vis))
        beams = next_beams

        if not beams:
            break

    return sorted(completed, key=lambda x: x[1], reverse=True)

def extend_subgraph_with_beam_search(
    kg: Dict[str, List[Dict[str, Any]]],
    current_triplets: List[Dict[str, Any]],
    question: str,
    summary_embeddings: Dict[str, np.ndarray],
    k: int = MAX_NEW_TRIPLETS,
    beam_width: int = BEAM_WIDTH,
    max_path_length: int = MAX_PATH_LENGTH
) -> List[Dict[str, Any]]:
    """Extend a subgraph using beam search to find relevant paths.
    
    Args:
        kg: Knowledge graph as adjacency list
        current_triplets: Current subgraph triplets
        question: Question to guide the search
        summary_embeddings: Dictionary mapping relation summaries to their embeddings
        k: Maximum number of new triplets to add
        beam_width: Width of beam search
        max_path_length: Maximum path length to search
        
    Returns:
        Extended list of triplets
    """
    # Get entities in current subgraph
    entities = {t["m"]["id"] for t in current_triplets} | {t["n"]["id"] for t in current_triplets}
    
    # Get question embedding
    question_emb = np.array(embeddings.embed_query(question))
    
    # Get existing triplet keys
    existing_keys = {(t["m"]["id"], t["r"]["id"], t["n"]["id"]) for t in current_triplets}
    
    # Find paths using beam search
    completed = beam_search_paths(
        kg,
        start_entities=entities,
        end_entities=entities,
        summary_embeddings=summary_embeddings,
        question_emb=question_emb,
        beam_width=beam_width,
        max_depth=max_path_length
    )
    
    if not completed:
        return current_triplets
        
    # Add new triplets
    selected = []
    selected_keys = set()
    for path, _ in completed:
        for triplet in path:
            key = (triplet["m"]["id"], triplet["r"]["id"], triplet["n"]["id"])
            if key not in existing_keys and key not in selected_keys:
                selected.append(triplet)
                selected_keys.add(key)
                if len(selected) >= k:
                    break
        if len(selected) >= k:
            break
            
    return current_triplets + selected

def add_triplet_context_to_question(question: str, check_relate: bool = False) -> str:
    """Add contextual information to a question based on relevant triplets.
    
    Args:
        question: Question to add context to
        check_relate: Whether to check each relation individually
        
    Returns:
        Question with added context if relevant triplets found
    """
    try:
        # Get initial relevant triplets
        relations = query_triplet_topk(question)
        
        if check_relate:
            check_rels = check_relations(question, relations)
            contextual_summary = contextual_question_retrieval(format_claim(check_rels)).summary if check_rels else ""
        else:
            try:
                # Get relevant triplets using batch check
                context = check_grade_lst(question, format_claim(relations)).passage_idx
                context = [int(x) for x in context.split(",")]
                check_rels = [relations[x-1] for x in context]
                
                # Build knowledge graph and extend subgraph using beam search
                kg = build_undirected_graph(relations)
                summary_embeddings = {
                    rel["r"]["summary"]: np.array(embeddings.embed_query(rel["r"]["summary"]))
                    for rel in relations
                }
                
                # Extend subgraph with beam search
                extended_rels = extend_subgraph_with_beam_search(
                    kg=kg,
                    current_triplets=check_rels,
                    question=question,
                    summary_embeddings=summary_embeddings
                )
                
                contextual_summary = contextual_question_retrieval(format_claim(extended_rels)).summary if extended_rels else ""
            except Exception as e:
                logger.warning(f"Error in batch relation check or beam search: {str(e)}")
                contextual_summary = ""
                
        if contextual_summary:
            question = f"{question} with some extra data: {contextual_summary}"
        return question
    except Exception as e:
        logger.error(f"Error adding context to question: {str(e)}")
        return question

def gen_cqr_triplet(query: str) -> str:
    """Generate contextual question retrieval for a query.
    
    Args:
        query: Query to generate CQR for
        
    Returns:
        Query with added context
    """
    return add_triplet_context_to_question(query, False)

def process_test_data() -> List[str]:
    """Process test data to generate CQR results.
    
    This function:
    1. Loads test data from CSV
    2. Processes each question in parallel
    3. Saves results to pickle file
    
    Returns:
        List of questions with added context
    """
    try:
        # Load test data
        logger.info("Loading test data...")
        df_test = pd.read_csv(TEST_DATA_PATH)
        test_data = df_test['question'].tolist()
        
        # Process in parallel
        logger.info(f"Processing {len(test_data)} questions using {NUM_PROCESSES} processes...")
        with Pool(NUM_PROCESSES) as pool:
            cqr_res = list(tqdm(
                pool.imap(gen_cqr_triplet, test_data),
                total=len(test_data)
            ))
        
        # Save results
        logger.info(f"Saving results to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump(cqr_res, f)
            
        return cqr_res
    except Exception as e:
        logger.error(f"Error processing test data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        process_test_data()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise