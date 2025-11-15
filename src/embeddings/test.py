import json
import requests
from typing import List
from tqdm import tqdm
from langchain.embeddings.base import Embeddings  # adjust if using another base class

class CustomAPIEmbeddings(Embeddings):
    def __init__(self, api_url: str, show_progress: bool = True, batch_size: int = 32):
        self.api_url = api_url
        self.show_progress = show_progress
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        lst_embedding = []
        iterator = range(0, len(texts), self.batch_size)
        iterator = tqdm(iterator) if self.show_progress else iterator

        for i in iterator:
            batch = texts[i: i + self.batch_size]
            payload = json.dumps({"inputs": batch})
            headers = {'Content-Type': 'application/json'}

            try:
                response = requests.post(self.api_url, headers=headers, data=payload)
                embeddings = json.loads(response.text)
                lst_embedding.extend(embeddings)  # assumes response is a list of embeddings
            except Exception as e:
                print(f"Error on batch {i // self.batch_size}: {e}")
                print(response.text if response else "No response")

        return lst_embedding

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]