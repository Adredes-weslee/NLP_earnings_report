"""
Embedding processor for NLP Earnings Report project.
Handles text embeddings using various models including transformers.
"""

import numpy as np
import pandas as pd
import logging
import os
import joblib
from typing import List, Dict, Union, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Optional imports for more advanced embedding techniques
try:
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger('embedding_processor')

class EmbeddingProcessor:
    """
    Class for creating and managing text embeddings using various techniques.
    Supports traditional bag-of-words, TF-IDF, and transformer-based embeddings.
    """
    
    def __init__(self, method: str = 'tfidf', model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding processor.
        
        Args:
            method: Embedding method ('bow', 'tfidf', 'transformer')
            model_name: Name of transformer model (if method='transformer')
        """
        self.method = method
        self.model_name = model_name
        self.vectorizer = None
        self.transformer_model = None
        self.tokenizer = None
        self.embedding_dim = None
        self.vocab = None
        
        logger.info(f"Initializing EmbeddingProcessor with method={method}")
        
        # Check if transformers are available when needed
        if method == 'transformer' and not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformer method selected but libraries not available. "
                          "Install with: pip install transformers sentence-transformers")
            raise ImportError("Transformers library not available")
    
    def fit(self, texts: List[str], max_features: int = 10000, **kwargs) -> 'EmbeddingProcessor':
        """
        Fit the embedding model on the provided texts.
        
        Args:
            texts: List of text documents
            max_features: Maximum vocabulary size for BoW/TF-IDF
            **kwargs: Additional arguments for the vectorizer
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.method} embedding model on {len(texts)} texts")
        
        if self.method == 'bow':
            self.vectorizer = CountVectorizer(max_features=max_features, **kwargs)
            self.vectorizer.fit(texts)
            self.vocab = self.vectorizer.get_feature_names_out()
            self.embedding_dim = len(self.vocab)
            
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
            self.vectorizer.fit(texts)
            self.vocab = self.vectorizer.get_feature_names_out()
            self.embedding_dim = len(self.vocab)
            
        elif self.method == 'transformer':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library not available")
                
            try:
                self.transformer_model = SentenceTransformer(self.model_name)
                # Get embedding dimension by encoding a sample text
                sample_embedding = self.transformer_model.encode(texts[0] if texts else "sample text")
                self.embedding_dim = sample_embedding.shape[0]
                logger.info(f"Loaded transformer model {self.model_name} with dimension {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Error loading transformer model: {str(e)}")
                raise
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to embeddings.
        
        Args:
            texts: List of text documents
            
        Returns:
            Embedding matrix of shape (n_samples, embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided to transform()")
            return np.array([])
            
        logger.info(f"Transforming {len(texts)} texts to embeddings")
        
        if self.method in ['bow', 'tfidf']:
            if self.vectorizer is None:
                raise ValueError("Model not fitted. Call fit() first.")
            return self.vectorizer.transform(texts)
            
        elif self.method == 'transformer':
            if self.transformer_model is None:
                raise ValueError("Model not fitted. Call fit() first.")
                
            embeddings = []
            batch_size = 32  # Process in batches to avoid memory issues
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.transformer_model.encode(batch_texts)
                embeddings.append(batch_embeddings)
                
            return np.vstack(embeddings)
    
    def fit_transform(self, texts: List[str], max_features: int = 10000, **kwargs) -> np.ndarray:
        """
        Fit the model and transform the texts in one step.
        
        Args:
            texts: List of text documents
            max_features: Maximum vocabulary size for BoW/TF-IDF
            **kwargs: Additional arguments for the vectorizer
            
        Returns:
            Embedding matrix
        """
        self.fit(texts, max_features, **kwargs)
        return self.transform(texts)
    
    def get_document_embeddings(self, df: pd.DataFrame, text_col: str) -> np.ndarray:
        """
        Get document embeddings for texts in a dataframe column.
        
        Args:
            df: DataFrame containing the texts
            text_col: Name of column containing text
            
        Returns:
            Document embedding matrix
        """
        texts = df[text_col].fillna('').tolist()
        return self.transform(texts)
    
    def save(self, path: str) -> None:
        """
        Save the embedding model to disk.
        
        Args:
            path: Directory path to save model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            'method': self.method,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        joblib.dump(config, os.path.join(path, 'config.joblib'))
        
        # Save model-specific components
        if self.method in ['bow', 'tfidf']:
            joblib.dump(self.vectorizer, os.path.join(path, 'vectorizer.joblib'))
        elif self.method == 'transformer':
            # For transformer models, we just save the model name as they're large
            # They will be reloaded from the HuggingFace hub
            with open(os.path.join(path, 'transformer_model.txt'), 'w') as f:
                f.write(self.model_name)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EmbeddingProcessor':
        """
        Load an embedding model from disk.
        
        Args:
            path: Directory path to load model from
            
        Returns:
            Loaded EmbeddingProcessor instance
        """
        config = joblib.load(os.path.join(path, 'config.joblib'))
        
        instance = cls(
            method=config['method'],
            model_name=config['model_name']
        )
        
        instance.embedding_dim = config['embedding_dim']
        
        # Load model-specific components
        if instance.method in ['bow', 'tfidf']:
            instance.vectorizer = joblib.load(os.path.join(path, 'vectorizer.joblib'))
            if hasattr(instance.vectorizer, 'get_feature_names_out'):
                instance.vocab = instance.vectorizer.get_feature_names_out()
            else:
                instance.vocab = instance.vectorizer.get_feature_names()
        elif instance.method == 'transformer':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library required but not available")
            # Load the model from HuggingFace hub
            instance.transformer_model = SentenceTransformer(instance.model_name)
        
        logger.info(f"Model loaded from {path}")
        return instance