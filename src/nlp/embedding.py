"""Embedding processor for NLP Earnings Report project.

This module provides utilities for converting raw text from earnings reports
into numerical vector representations (embeddings) that can be used for 
downstream machine learning tasks. It supports multiple embedding methods:

- Bag-of-Words: Simple count-based document representations
- TF-IDF: Term frequency-inverse document frequency weighting
- Transformer models: Contextual embeddings from neural network models

The main class, EmbeddingProcessor, handles the creation, storage, and 
application of these embedding models in a unified interface.
"""

import numpy as np
import pandas as pd
import logging
import os
import joblib
import sys
from typing import List, Dict, Union, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Import configuration values
from ..config import (MAX_FEATURES, NGRAM_RANGE, MAX_DOC_FREQ, EMBEDDING_MODEL_PATH,
                  MODEL_DIR, RANDOM_STATE)

# Optional imports for more advanced embedding techniques
try:
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger('embedding_processor')

class EmbeddingProcessor:
    """Class for creating and managing text embeddings using various techniques.
    
    This class provides tools to convert financial text documents into numerical
    vector representations using different embedding techniques. It supports 
    traditional methods like bag-of-words and TF-IDF, as well as modern 
    transformer-based embeddings when available.
    
    Attributes:
        method (str): The embedding method being used.
        model_name (str): The name of the transformer model if applicable.
        vectorizer: The vectorizer object for traditional embeddings.
        transformer_model: The loaded transformer model if applicable.
        tokenizer: The tokenizer associated with the transformer model.
        embedding_dim (int): Dimensionality of the embedding vectors.
        vocab (list): Vocabulary used for traditional embedding methods.
    """
    
    def __init__(self, method: str = 'tfidf', model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding processor with specified method and model.
        
        Args:
            method (str): Embedding method to use. Options are:
                'bow': Bag-of-Words representation
                'tfidf': Term Frequency-Inverse Document Frequency
                'transformer': Neural transformer-based embeddings
                Defaults to 'tfidf'.
            model_name (str): Name of transformer model to use if method is 'transformer'.
                Defaults to 'all-MiniLM-L6-v2', a good balance of quality and speed.
                
        Example:
            >>> processor = EmbeddingProcessor(method='tfidf')
            >>> processor.fit(documents)
            >>> embeddings = processor.transform(new_documents)
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
        
    def fit(self, texts: List[str], max_features: int = MAX_FEATURES, **kwargs) -> 'EmbeddingProcessor':
        """Fit the embedding model on the provided texts.
        
        This method trains the embedding model on a corpus of documents.
        For bag-of-words and TF-IDF, it builds the vocabulary and calculates
        document frequencies. For transformer-based methods, it loads the
        pre-trained model.
        
        Args:
            texts (List[str]): List of text documents to fit the model on.
            max_features (int, optional): Maximum vocabulary size for BoW/TF-IDF.
                Defaults to MAX_FEATURES from config.
            **kwargs: Additional arguments for the vectorizer or transformer model.
                Common options include:
                - ngram_range: Tuple specifying n-gram range
                - stop_words: List of stop words or 'english'
                - max_df: Maximum document frequency threshold
            
        Returns:
            EmbeddingProcessor: Self for method chaining.
            
        Raises:
            ValueError: If the texts provided are empty or not in the expected format.
            
        Example:
            >>> processor = EmbeddingProcessor(method='tfidf')
            >>> processor.fit(training_documents)
        """
        logger.info(f"Fitting {self.method} embedding model on {len(texts)} texts")
          # Prepare common kwargs with config values if not already provided
        common_kwargs = {
            'max_features': max_features,
            'ngram_range': NGRAM_RANGE,
            'max_df': MAX_DOC_FREQ
        }
        # Update with any user-provided kwargs
        common_kwargs.update(kwargs)
        
        if self.method == 'bow':
            self.vectorizer = CountVectorizer(**common_kwargs)
            self.vectorizer.fit(texts)
            self.vocab = self.vectorizer.get_feature_names_out()
            self.embedding_dim = len(self.vocab)
            
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**common_kwargs)
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
        """Transform texts to embeddings using the fitted model.
        
        This method converts input texts to numerical embeddings using the
        previously fit model. The output format depends on the embedding method:
        - For 'bow' and 'tfidf': Returns sparse matrices
        - For 'transformer': Returns dense matrices
        
        Args:
            texts (List[str]): List of text documents to transform into embeddings.
            
        Returns:
            np.ndarray: Embedding matrix of shape (n_samples, embedding_dim)
                where n_samples is the number of input texts and embedding_dim
                is the dimensionality of the embedding space.
                
        Raises:
            ValueError: If the model hasn't been fitted yet.
            
        Example:
            >>> processor = EmbeddingProcessor(method='tfidf')
            >>> processor.fit(training_documents)
            >>> embeddings = processor.transform(test_documents)
            >>> print(f"Embedding shape: {embeddings.shape}")
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
        """Fit the model and transform the texts in one step.
        
        This is a convenience method that calls fit() followed by transform()
        in a single operation. It's equivalent to calling these methods separately
        but may be more efficient for some embedding methods.
        
        Args:
            texts (List[str]): List of text documents to fit on and transform.
            max_features (int, optional): Maximum vocabulary size for BoW/TF-IDF.
                For 'bow' and 'tfidf' methods, this limits the size of the vocabulary.
                Higher values capture more terms but increase dimensionality.
                Defaults to 10000.
            **kwargs: Additional arguments for the vectorizer or transformer.
                Common options include:
                - ngram_range: Tuple specifying n-gram range, e.g. (1, 2) for unigrams and bigrams
                - min_df: Minimum document frequency threshold for terms
                - stop_words: List of stop words or 'english' to use built-in list
            
        Returns:
            np.ndarray: Embedding matrix of shape (n_samples, embedding_dim).
                For 'bow' and 'tfidf', this is typically a sparse matrix.
                For 'transformer', this is a dense matrix.
            
        Example:
            >>> processor = EmbeddingProcessor(method='tfidf')
            >>> embeddings = processor.fit_transform(documents)
            >>> print(f"Embedding shape: {embeddings.shape}")
            
        Note:
            For large datasets with 'transformer' method, this operation may be
            memory-intensive. Consider using fit() and transform() separately with
            batching for very large datasets.
        """
        self.fit(texts, max_features, **kwargs)
        return self.transform(texts)
    
    def get_document_embeddings(self, df: pd.DataFrame, text_col: str) -> np.ndarray:
        """Get document embeddings for texts in a dataframe column.
        
        This is a convenience method that extracts text from a dataframe column
        and transforms it into embeddings. It handles null values by replacing 
        them with empty strings.
        
        Args:
            df (pd.DataFrame): DataFrame containing the texts to embed.
            text_col (str): Name of the column containing the text data.
            
        Returns:
            np.ndarray: Document embedding matrix with shape (len(df), embedding_dim).
            
        Raises:
            KeyError: If the specified text_col doesn't exist in the dataframe.
            
        Example:
            >>> processor = EmbeddingProcessor(method='tfidf')
            >>> processor.fit(training_documents)
            >>> embeddings = processor.get_document_embeddings(test_df, 'text_column')
        """
        texts = df[text_col].fillna('').tolist()
        return self.transform(texts)    
    
    def save(self, path: str = None) -> None:
        """Save the embedding model to disk.
        
        This method serializes the embedding model and its configuration
        to the specified directory. For traditional models like TF-IDF,
        it saves the vectorizer. For transformer models, it saves the
        model configuration and a reference to the pre-trained model.
        
        Args:
            path (str, optional): Directory path to save the model.
                If None, uses EMBEDDING_MODEL_PATH from config.
                
        Raises:
            OSError: If there's an error creating the directory or writing files.
            
        Example:
            >>> processor = EmbeddingProcessor(method='tfidf')
            >>> processor.fit(documents)
            >>> processor.save('models/embeddings/tfidf_model/')
        """
        if path is None:
            path = EMBEDDING_MODEL_PATH
            
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
    def load(cls, path: str = None) -> 'EmbeddingProcessor':
        """Load an embedding model from disk.
        
        This class method reconstructs an EmbeddingProcessor instance from
        saved files. It loads the configuration and model components from
        the specified directory.
        
        Args:
            path (str, optional): Directory path to load the model from.
                If None, uses EMBEDDING_MODEL_PATH from config.
            
        Returns:
            EmbeddingProcessor: A loaded instance with the saved configuration
                and model components.
                
        Raises:
            FileNotFoundError: If the model files cannot be found at the specified path.
            
        Example:
            >>> processor = EmbeddingProcessor.load('models/embeddings/tfidf_model/')
            >>> embeddings = processor.transform(new_documents)
        """
        if path is None:
            path = EMBEDDING_MODEL_PATH
            
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