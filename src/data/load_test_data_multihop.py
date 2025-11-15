import json
from util import rm_file
from tqdm import tqdm 
import argparse
from copy import deepcopy
import os 
from util import JSONReader
import openai
import pickle
from typing import List, Dict
from tqdm import tqdm

from llama_index import (
	ServiceContext,
	OpenAIEmbedding,
	PromptHelper,
	VectorStoreIndex,
	set_global_service_context
)
from llama_index.extractors import BaseExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding,VoyageEmbedding,InstructorEmbedding
from llama_index.postprocessor import FlagEmbeddingReranker
from llama_index.schema import QueryBundle, MetadataMode

from langchain_core.documents import Document


### Embeeding

### Call API Endpoint Embedding
import json
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from tqdm.notebook import tqdm

class CustomAPIEmbeddings(Embeddings):
    def __init__(self, api_url: str, show_progress:bool):  # model_name: strKG_enhance_RAG_Finance_News
        # self.model_name = model_name
        self.api_url = api_url
        self.show_progress = show_progress

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        lst_embedding = []
        if self.show_progress: # for tqdm embedding
            for query in tqdm(texts):
                payload = json.dumps({
                  "query": query
                })
                headers = {
                  'Content-Type': 'application/json'
                }
                try:
                    response = json.loads(requests.request("POST", self.api_url, headers=headers, data=payload).text)['embedding']
                except:
                    print(requests.request("POST", self.api_url, headers=headers, data=payload).text)
                lst_embedding.append(response)
        else:
            for query in texts:
                payload = json.dumps({
                  "query": query
                })
                headers = {
                  'Content-Type': 'application/json'
                }
                try:
                    response = json.loads(requests.request("POST", self.api_url, headers=headers, data=payload).text)['embedding']
                except:
                    print(requests.request("POST", self.api_url, headers=headers, data=payload).text)
                lst_embedding.append(response)
            
        return lst_embedding  # Adjust this based on the response format of your API

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
		
class CustomExtractor(BaseExtractor):
	async def aextract(self, nodes) -> List[Dict]:
		metadata_list = [
			{
				"title": (
					node.metadata["title"]
				),
				"source": (
					node.metadata["source"]
				),      
				"published_at": (
					node.metadata["published_at"]
				)
			}
			for node in nodes
		]
		return metadata_list
	
if __name__ == '__main__':
	openai.api_key = "test"
	openai.base_url = "http://127.0.0.1:9012/v1/"
	
	parser = argparse.ArgumentParser(description="running script.")
	parser.add_argument('--retriever_url', type=str, required=True, help='retriever name')
	parser.add_argument('--output_dir', type=str, required=True, help='output path')
	parser.add_argument('--llm', type=str, required=False,default="Meta-Llama-3-70B-Instruct", help='LLMs')
	parser.add_argument('--rerank', action='store_true',required=False,default=False, help='if rerank')
	parser.add_argument('--topk', type=int, required=False,default=10, help='Top K')
	parser.add_argument('--chunk_size', type=int, required=False,default=512, help='chunk_size')
	parser.add_argument('--context_window', type=int, required=False,default=4096, help='context_window')
	parser.add_argument('--num_output', type=int, required=False,default=512, help='num_output')

	args = parser.parse_args()
	rerank = args.rerank
	top_k = args.topk
	llm = OpenAI(model=args.llm, temperature=0, max_tokens=args.context_window)
	embeddings = CustomAPIEmbeddings(api_url=args.retriever_url, show_progress=False)

	# service context 
	text_splitter = SentenceSplitter(chunk_size=args.chunk_size)
	prompt_helper = PromptHelper(
		context_window=args.context_window,
		num_output=args.num_output,
		chunk_overlap_ratio=0.0,
		chunk_size_limit=None,
	)
	service_context = ServiceContext.from_defaults(
		llm=llm,
		embed_model=embeddings,
		text_splitter=text_splitter,
		prompt_helper=prompt_helper,
	)
	set_global_service_context(service_context)

	reader = JSONReader()
	data = reader.load_data('/home/minhb/doan/KG-Contextual-Question-Retrieval/src/data/raw_data/multihopRAG/corpus.json')
	print(data[0])
	raw_content = [x.text for x in data]
	# print(data[0])

		
	transformations = [text_splitter,CustomExtractor()] 
	pipeline = IngestionPipeline(transformations=transformations)
	nodes = pipeline.run(documents=data)
	nodes_see = deepcopy(nodes)
	print(
		"LLM sees:\n",
		(nodes_see)[0].get_content(metadata_mode=MetadataMode.LLM),
	)
	print('Finish Loading...')

	print(len(nodes), nodes[0], type(nodes))
	lst_chunks = []
	for node in tqdm(nodes):
		metadata = node.metadata
		for doc in raw_content:
			if node.text in doc:
				metadata['raw_content'] = doc
				metadata['chunks_content'] = node.text
				break
		if 'raw_content' not in metadata.keys():
			print("Doc not found")
		document = Document(
		    page_content=node.text,
		    metadata=metadata
		)
		lst_chunks.append(document)
		
		
	with open(args.output_dir,"wb") as f:
		pickle.dump(lst_chunks, f)
