"""
Embeddings generation module using Sentence Transformers
We use all-MiniLM-L6-v2 model which is:
- Small and fast (good for this demo)
- Produces 384-dimensional embeddings
- Works well for semantic similarity
- Runs locally (no API limits)
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import faiss


class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding model"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 embedding size

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Create FAISS index using Inner Product (Cosine Similarity)
        Embeddings must be L2-normalized before indexing
        """
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        return index


class VectorStore:
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.index = None
        self.chunks = []
        self.chunk_texts = []

    def add_documents(self, chunks: List[dict]):
        """Add document chunks to vector store"""
        self.chunks = chunks
        self.chunk_texts = [chunk['text'] for chunk in chunks]

        print("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(self.chunk_texts)

        print("Building FAISS index...")
        self.index = self.embedding_generator.create_faiss_index(embeddings)
        print(f"Index built with {len(chunks)} chunks")

    def search(self, query: str, k: int = 3) -> List[Tuple[dict, float]]:
        """Search for top-k relevant chunks using cosine similarity"""
        if self.index is None:
            raise ValueError("Index not initialized. Call add_documents() first.")

        # Generate and normalize query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])

        # Perform similarity search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results
