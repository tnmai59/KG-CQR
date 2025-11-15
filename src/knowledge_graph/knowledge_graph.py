import argparse
import os
import json
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.runnables import RunnableConfig
from langchain_core.documents.base import Document
import pickle
import copy
from multiprocessing import Pool
from tqdm import tqdm
import asyncio
import llm_utils
from pydantic import BaseModel, Field, create_model


class CustomAPIEmbeddings(Embeddings):
	def __init__(self, api_url: str):
		self.api_url = api_url

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		lst_embedding = []
		for query in texts:
			payload = json.dumps({"query": query})
			headers = {"Content-Type": "application/json"}
			try:
				response = json.loads(
					requests.request("POST", self.api_url, headers=headers, data=payload).text
				)["embedding"]
			except Exception as e:
				print(f"Error embedding query: {e}")
				response = []
			lst_embedding.append(response)
		return lst_embedding

	def embed_query(self, text: str) -> List[float]:
		return self.embed_documents([text])[0]


def postprocess_graph(graph):
	"""Postprocess graph to clean and adjust node/relationship properties."""
	lst_graph_filter_none = [i for i in graph if len(i.nodes) > 0]
	dct_map = {}

	for i in lst_graph_filter_none:
		for node in i.nodes:
			node.properties['node_type'] = copy.deepcopy(node.type)
			node.type = "Node"
			if node.id not in dct_map:
				dct_map[node.id] = node.properties['node_type']
			else:
				node.properties['node_type'] = dct_map[node.id]

	for i in lst_graph_filter_none:
		for relationship in i.relationships:
			for entity in [relationship.source, relationship.target]:
				if entity.id not in dct_map:
					dct_map[entity.id] = copy.deepcopy(entity.type)
					entity.type = "Node"
					entity.properties['node_type'] = dct_map[entity.id]
				else:
					entity.type = "Node"
					entity.properties['node_type'] = dct_map[entity.id]
	return lst_graph_filter_none

def process(content):
	global llm
	global llm_transformer
	return llm(content)
	# structured_llm = llm.with_structured_output(llm_transformer.schema, include_raw=True)
	# return structured_llm.invoke(content)

def process_graph_async(args, llm_transformer, lst_docs):
	"""Process graph asynchronously."""
	cnt = 0
	# cfg = RunnableConfig(max_concurrency=args.num_proc)
	# Get prompt doc
	print("Get prompt docs")
	prompt_docs = [llm_transformer.get_prompt(docs) for docs in tqdm(lst_docs)]
	# print(prompt_docs[0])
	with Pool(args.num_proc) as pool:
		results = list(tqdm(pool.imap(process, prompt_docs), total=len(prompt_docs), desc="Extracting..."))
	# print(results)
	graph = []
	# Postprocessing langchain graph
	print("Running postprocessing langchain graph")
	for i, j in tqdm(zip(lst_docs, results)):
		try:
			graph.append(llm_transformer.postprocess(i, j))
		except:
			cnt += 1
	print("Process error : ", cnt)
	return postprocess_graph(graph)

def custom_process_graph_async(args, llm_transformer, lst_docs):
	"""Process graph asynchronously."""
	graph = llm_transformer.custom_convert_to_graph_documents(lst_docs)
	return postprocess_graph(graph)

def process_graph_sync(args, llm_transformer, lst_docs):
	"""Process graph synchronously."""
	graph = llm_transformer.convert_to_graph_documents(lst_docs)
	return postprocess_graph(graph)

async def async_process_document(doc):
	global llm_transformer
	return await llm_transformer.aprocess_response(doc)

# Wrapper for running asyncio in a multiprocessing worker
def process_document_worker(doc):
    return asyncio.run(async_process_document(doc))

# Function to use multiprocessing with tqdm
def process_documents_with_multiprocessing(documents, num_workers=4):
    with Pool(num_workers) as pool:
        results = []
        for result in tqdm(pool.imap(process_document_worker, documents), total=len(documents), desc="Processing"):
            results.append(result)
    return results


def main():
	global llm
	global llm_transformer
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, required=True, default="Meta-Llama-3-70B-Instruct")
	parser.add_argument("--inference_server_url", type=str, required=False)
	parser.add_argument("--openai_api_key", type=str, required=True)
	parser.add_argument("--input_file", type=str, required=True)
	parser.add_argument("--output_file", type=str, required=True)
	parser.add_argument("--use_sample", action="store_true")
	parser.add_argument("--proxy", type=str, required=False, default="")
	parser.add_argument("--use_async", action="store_true")
	parser.add_argument('--num_proc', type=int, default=4, help='Number of processes')
	args = parser.parse_args()

	# Proxy Configuration
	os.environ["http_proxy"] = args.proxy
	os.environ["https_proxy"] = args.proxy

	# LLM Configuration
	if args.inference_server_url:
		llm = ChatOpenAI(
			model=args.model_name, openai_api_key=args.openai_api_key, openai_api_base=args.inference_server_url
		)
	else:
		llm = ChatOpenAI(model=args.model_name, openai_api_key=args.openai_api_key)

	# Load input documents
	with open(args.input_file, "rb") as f:
		lst_docs = pickle.load(f)

	if args.use_sample:
		lst_docs = lst_docs[:100]

	llm_transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=False)

	# Process graph
	print("Start calling graph")
	if args.use_async:
		# graph = process_graph_async(args, llm_transformer, lst_docs)
		# graph = custom_process_graph_async(args, llm_transformer, lst_docs)
		# graph = llm_transformer.aconvert_to_graph_documents(lst_docs)
		graph = process_documents_with_multiprocessing(lst_docs, args.num_proc)
	else:
		graph = process_graph_sync(args, llm_transformer, lst_docs)
	# print(graph)
	# Save graph to output file
	with open(args.output_file, "wb") as f:
		pickle.dump(graph, f)


if __name__ == "__main__":
	main()
