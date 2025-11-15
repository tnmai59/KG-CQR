"""Module for handling embeddings functionality."""

import json
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from tqdm.notebook import tqdm

from .config import EMBEDDING_API_URL

class CustomAPIEmbeddings(Embeddings):
    """Custom embeddings class that uses an API endpoint for generating embeddings."""
    
    def __init__(self, api_url: str = EMBEDDING_API_URL, show_progress: bool = False):
        """Initialize the embeddings class.
        
        Args:
            api_url: URL of the embedding API endpoint
            show_progress: Whether to show progress bar during embedding generation
        """
        self.api_url = api_url
        self.show_progress = show_progress

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        lst_embedding = []
        texts_iter = tqdm(texts) if self.show_progress else texts
        
        for query in texts_iter:
            payload = json.dumps({"query": query})
            headers = {'Content-Type': 'application/json'}
            
            try:
                response = requests.post(self.api_url, headers=headers, data=payload)
                response.raise_for_status()
                embedding = json.loads(response.text)['embedding']
                lst_embedding.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                print(f"Response: {response.text if 'response' in locals() else 'No response'}")
                raise
                
        return lst_embedding

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0] 