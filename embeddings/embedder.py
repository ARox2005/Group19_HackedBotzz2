"""
Embedding module for generating vector embeddings from text.
Uses Sentence Transformers for high-quality text embeddings.
"""

import os
from typing import List, Optional, Union
import numpy as np

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Text embedding model using Sentence Transformers.
    Supports GPU/CPU fallback and batch processing.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the Sentence Transformer model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading embedding model '{model_name}' on {self.device}...")
        
        # Load the model
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_dir
        )
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"Embedding model loaded. Dimension: {self.embedding_dim}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        return embeddings
    
    def embed_documents(
        self,
        documents: List[dict],
        text_key: str = 'text',
        batch_size: int = 32
    ) -> List[dict]:
        """
        Add embeddings to document dictionaries.
        
        Args:
            documents: List of document dictionaries
            text_key: Key containing text to embed
            batch_size: Batch size for processing
            
        Returns:
            Documents with 'embedding' key added
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc.get(text_key, '') for doc in documents]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        return documents
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    def to(self, device: str) -> 'EmbeddingModel':
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self
