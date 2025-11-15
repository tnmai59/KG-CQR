from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel
from multiprocessing import Pool, Manager
import pickle
import faiss
import traceback
import uuid

from utils import *
from config import *
from struct_output import *
from beam import *

from agent import *
from struct_output import *


llm = ChatOpenAI(
	model=MODEL_NAME,
	openai_api_key="EMPTY",
	max_tokens=512,
	openai_api_base=INFERENCE_SERVER_URL,
	temperature=0,
	streaming= False
)

embeddings = CustomAPIEmbeddings()
dct_mapping_triplet, lst_embedding = load_triplet_data(TRIPLET_MAP_PATH, TRIPLET_EMB_PATH)
index = initialize_faiss_index(lst_embedding)
lst_chunks = load_corpus(CORPUS_PATH)

# Initialize BM25
tokenized_corpus = [doc.split(" ") for doc in lst_chunks]
bm25 = BM25Okapi(tokenized_corpus)


def retrieval_bm25(question, k):
	tokenized_query = question.split(" ")
	lst_retrieval = bm25.get_top_n(tokenized_query, lst_chunks, n=k)
	return lst_retrieval

# Initialize BGE
embeddings_doc = embeddings.embed_documents(lst_chunks)
p_embd = []
for i in tqdm(range(len(embeddings_doc))):
	p_embd.append(embeddings_doc[i])
p_embd = np.array(p_embd)
index_p = faiss.IndexFlatIP(p_embd.shape[1])  # IP = Inner Product for cosine similarity
index_p.add(p_embd)

def retrieval_bge(query, k, alpha=0.7):
	query = query.split(" with some extra data: ")
	if len(query) > 1:
		question = query[0]
		context = query[1]
		q_embd = np.array(embeddings.embed_query(max_len(question)))
		if len(context) > 0:
			c_embd = np.array(embeddings.embed_query(max_len(context)))
			v_fuse = alpha*q_embd + (1-alpha)*c_embd
			v_fuse = v_fuse.reshape(1, -1)
		else:
			v_fuse = q_embd.reshape(1,-1)
	else:
		question = query[0]
		q_embd = np.array(embeddings.embed_query(max_len(question)))
		v_fuse = q_embd.reshape(1,-1)
	distances, indices = index_p.search(v_fuse, k)
	indices = indices[0]
	retrieved_docs = []
	for idx in indices:
		retrieved_docs.append(lst_chunks[idx])
	return retrieved_docs



def process_question(tasks):
	"""Process a single question."""
	question, label, k, n_loop, qid, retrieval_mode = tasks  # Unpack the arguments
	try:
		i = 0
		thought_q = ""
		pt = []
		gen_answer = None  # Ensure it's always defined
		if retrieval_mode == "bge":
			context = max_length_context(retrieval_bge(question, k))
		if retrieval_mode == "bm25":
			context = max_length_context(retrieval_bm25(question, k))
		while i < n_loop:
			check = check_response(question, format_docs(context)).binary_score
			if check or (not check and i == n_loop - 1):
				gen_answer = final_answer(question, format_docs(context))
				break
			else:
				new_CoT_query = gen_question(question, format_docs(context), "\n".join(pt)).new_query
				pt.append(new_CoT_query)
				thought_q += f"\n{i}-{new_CoT_query}"
				if retrieval_mode == "bge":
					new_context = max_length_context(retrieval_bge(new_CoT_query, k))
				if retrieval_mode == "bm25":
					new_context = max_length_context(retrieval_bm25(new_CoT_query, k))
				context = list(set(context + new_context))  # Deduplicate
			i += 1

		res = {
			"Question": question,
			"id": qid,
			"Answer": gen_answer,
			"Label": label,
			"Context": context,
			"CoT": thought_q,
			"n_CoT": int(i),
		}
	except Exception as e:
		print(f"Error occurred during processing question '{question}': {e}")
		traceback.print_exc()
		res = None

	fn = uuid.uuid4()
	with open(f"{fn}.pkl", "wb") as f:
		pickle.dump(res, f)  # Corrected from dumping `fn` to dumping `res`
	return res



if __name__ == "__main__":
	k = 8 
	n_loop = 5 
	num_procs = 8
	df_test = pd.read_json(TEST_DATA_PATH, lines=True) 
	questions = df_test['question'].tolist()
	labels = df_test["response"].tolist()
	ids = df_test["id"].tolist()
	tasks = [(questions[i], labels[i], k, n_loop, ids[i], "bm25") for i in range(len(questions))]

	# Use a Manager list to store results
	with Manager() as manager:
		with Pool(20) as pool:
			results = list(tqdm(pool.imap(process_question, tasks), total=len(tasks)))

		results = [res for res in results if res is not None]
		final_test = pd.DataFrame(results)

		# Save to an Excel file
		final_test.to_excel("output.xlsx", index=False)
		print("Processing complete.")
