"""Unified NLP processing module for financial text analysis.

This module provides centralized text processing functionality for the entire
NLP Earnings Report pipeline. It handles vectorization, common NLP tasks,
and serves as the single source of truth for text representations.

The module centralizes:
- Document-term matrix creation
- Vocabularies and tokenization rules
- Shared embedding logic
- Basic text statistics

By consolidating these functions, it eliminates duplication across the pipeline
and ensures consistency in text processing across all components.
"""

import numpy as np
import pandas as pd
import re
import os
import logging
from typing import List, Dict, Union, Optional, Tuple, Any
import pickle
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

# Import configuration values
from ..config import (MAX_FEATURES, NGRAM_RANGE, MAX_DOC_FREQ, 
                      RANDOM_STATE, MODEL_DIR)

logger = logging.getLogger('nlp_processing')

class NLPProcessor:
    """Central processing class for all NLP operations in the financial analysis pipeline.
    
    This class serves as the core processor for text data across the NLP pipeline,
    handling common operations like vectorization, text statistics, and coordinating
    between different components like embedding, topic modeling, and sentiment analysis.
    
    By centralizing text processing, it ensures consistent tokenization, vocabulary,
    and vector representations across all pipeline components.
    
    Attributes:
        count_vectorizer (CountVectorizer): Vectorizer for creating document-term matrices
        tfidf_vectorizer (TfidfVectorizer): Vectorizer for TF-IDF weighted representations 
        vocab (list): Shared vocabulary across vectorizers
    """
    
    def __init__(self, max_features: int = MAX_FEATURES, 
                 ngram_range: Tuple[int, int] = NGRAM_RANGE,
                 max_df: float = MAX_DOC_FREQ,
                 random_state: int = RANDOM_STATE):
        """Initialize the NLP processor with configuration settings.
        
        Args:
            max_features (int, optional): Maximum vocabulary size for vectorizers.
                Defaults to MAX_FEATURES from config.
            ngram_range (tuple, optional): Range of n-grams to include in vocabulary.
                For example, (1, 2) means unigrams and bigrams. Defaults to NGRAM_RANGE.
            max_df (float, optional): Maximum document frequency threshold (0.0-1.0).
                Terms with higher document frequency will be filtered out.
                Defaults to MAX_DOC_FREQ from config.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to RANDOM_STATE from config.
                
        Example:
            >>> processor = NLPProcessor(max_features=5000)
            >>> # Get document-term matrix
            >>> dtm, vocab = processor.create_document_term_matrix(texts)
            >>> # Get TF-IDF matrix using the same vocabulary
            >>> tfidf_matrix = processor.create_tfidf_matrix(texts)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.random_state = random_state
        
        # Initialize vectorizers as None, they'll be created on demand
        self.count_vectorizer = None
        self.tfidf_vectorizer = None
        self.vocab = None
        
        # Initialize stopwords
        self.stop_words = list(set(stopwords.words('english')))
        # # Add financial stopwords
        # self.financial_stopwords = {'company', 'quarter', 'year', 'financial', 'reported', 'period',
        #                           'quarter', 'fiscal', 'results', 'earnings', 'reports', 'press', 
        #                           'release', 'corporation', 'announces', 'announced', 'today'}
        # self.stop_words.update(self.financial_stopwords)
        
        logger.info(f"Initialized NLPProcessor with max_features={max_features}")
        
    def create_document_term_matrix(self, texts: List[str], 
                                   fit: bool = True,
                                   use_existing: bool = True,
                                   save_path: str = None) -> Tuple:
        """Create a document-term matrix from texts using CountVectorizer.
        
        This is the central method for creating document-term matrices throughout
        the pipeline. It creates a consistent vocabulary and ensures the same
        tokenization rules are applied everywhere.
        
        Args:
            texts (List[str]): List of text documents to analyze
            fit (bool, optional): Whether to fit the vectorizer on these texts.
                When False, uses a previously fit vectorizer. Defaults to True.
            use_existing (bool, optional): Whether to use an existing vectorizer
                if one has already been created. Defaults to True.
            save_path (str, optional): Path to save the fitted vectorizer.
                If None, the vectorizer is not saved. Defaults to None.
                
        Returns:
            Tuple: A tuple containing:
                - document-term matrix (sparse matrix)
                - list of vocabulary terms (feature names)
                
        Example:
            >>> processor = NLPProcessor()
            >>> dtm, vocab = processor.create_document_term_matrix(texts)
            >>> print(f"Matrix shape: {dtm.shape}, Vocabulary size: {len(vocab)}")
            
        Note:
            This method is designed to be the single source of truth for
            document-term matrices throughout the pipeline. Other components
            should use this method rather than creating their own vectorizers.
        """
        # Use existing vectorizer if available and requested
        if use_existing and self.count_vectorizer is not None and not fit:
            logger.info("Using existing count vectorizer")
            dtm = self.count_vectorizer.transform(texts)
            return dtm, self.vocab
        
        # Create a new vectorizer if fitting or no existing vectorizer
        if fit or self.count_vectorizer is None:
            logger.info(f"Creating new CountVectorizer with max_features={self.max_features}")
            
            # Initialize the vectorizer with stop words
            self.count_vectorizer = CountVectorizer(
                token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b',
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                stop_words=self.stop_words,
                max_df=self.max_df
            )
            
            # Fit and transform
            dtm = self.count_vectorizer.fit_transform(texts)
            self.vocab = self.count_vectorizer.get_feature_names_out()
        else:
            # Transform only
            dtm = self.count_vectorizer.transform(texts)
        
        logger.info(f"DTM shape: {dtm.shape}")
        
        # Save the vectorizer if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.count_vectorizer, f)
            logger.info(f"Count vectorizer saved to {save_path}")
        
        return dtm, self.vocab
    
    def create_tfidf_matrix(self, texts: List[str], 
                           fit: bool = True,
                           use_existing: bool = True,
                           save_path: str = None) -> Tuple:
        """Create a TF-IDF matrix from texts using TfidfVectorizer.
        
        This method creates TF-IDF weighted document vectors using consistent
        tokenization rules with the count vectorizer. It's useful for tasks that
        benefit from term weighting like information retrieval or text classification.
        
        Args:
            texts (List[str]): List of text documents to analyze
            fit (bool, optional): Whether to fit the vectorizer on these texts.
                When False, uses a previously fit vectorizer. Defaults to True.
            use_existing (bool, optional): Whether to use an existing vectorizer
                if one has already been created. Defaults to True.
            save_path (str, optional): Path to save the fitted vectorizer.
                If None, the vectorizer is not saved. Defaults to None.
                
        Returns:
            Tuple: A tuple containing:
                - TF-IDF matrix (sparse matrix)
                - list of vocabulary terms (feature names)
                
        Example:
            >>> processor = NLPProcessor()
            >>> tfidf_matrix, vocab = processor.create_tfidf_matrix(texts)
            >>> print(f"Matrix shape: {tfidf_matrix.shape}")
            
        Note:
            This method uses the same tokenization and vocabulary settings as
            create_document_term_matrix() to ensure consistency throughout the pipeline.
        """
        # Use existing vectorizer if available and requested
        if use_existing and self.tfidf_vectorizer is not None and not fit:
            logger.info("Using existing TF-IDF vectorizer")
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            return tfidf_matrix, self.vocab
        
        # Create a new vectorizer if fitting or no existing vectorizer
        if fit or self.tfidf_vectorizer is None:
            logger.info(f"Creating new TfidfVectorizer with max_features={self.max_features}")
            
            self.tfidf_vectorizer = TfidfVectorizer(
                token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b',
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                stop_words=self.stop_words,
                max_df=self.max_df
            )
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.vocab = self.tfidf_vectorizer.get_feature_names_out()
        else:
            # Transform only
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Save the vectorizer if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            logger.info(f"TF-IDF vectorizer saved to {save_path}")
        
        return tfidf_matrix, self.vocab
    
    def get_vectorized_text(self, texts: List[str], method: str = 'count') -> Tuple:
        """Get vectorized text representation using the specified method.
        
        This is a convenience method that wraps the document-term matrix and
        TF-IDF matrix creation methods. It's useful when you need a specific
        type of representation but don't care about the implementation details.
        
        Args:
            texts (List[str]): List of text documents to vectorize
            method (str, optional): Vectorization method to use:
                - 'count': Standard document-term count matrix
                - 'tfidf': TF-IDF weighted matrix
                Defaults to 'count'.
                
        Returns:
            Tuple: A tuple containing:
                - The vectorized matrix (sparse matrix)
                - List of vocabulary terms (feature names)
                
        Raises:
            ValueError: If an unknown method is specified
            
        Example:
            >>> processor = NLPProcessor()
            >>> matrix, vocab = processor.get_vectorized_text(texts, method='tfidf')
        """
        if method == 'count':
            return self.create_document_term_matrix(texts)
        elif method == 'tfidf':
            return self.create_tfidf_matrix(texts)
        else:
            raise ValueError(f"Unknown vectorization method: {method}")
    
    def compute_text_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Compute statistical metrics about a collection of texts.
        
        This method calculates various metrics like average length, vocabulary size,
        and other statistics useful for understanding the text corpus characteristics.
        
        Args:
            texts (List[str]): List of text documents to analyze
            
        Returns:
            Dict[str, float]: Dictionary with text statistics including:
                - mean_length: Average word count per document
                - median_length: Median word count across documents
                - total_documents: Number of documents analyzed
                - vocabulary_size: Number of unique terms
                - density: Average non-zero terms per document
                
        Example:
            >>> processor = NLPProcessor()
            >>> stats = processor.compute_text_statistics(texts)
            >>> print(f"Average document length: {stats['mean_length']:.1f} words")
        """
        # Calculate length statistics
        text_lengths = [len(text.split()) for text in texts]
        
        stats = {
            'mean_length': np.mean(text_lengths),
            'median_length': np.median(text_lengths),
            'min_length': min(text_lengths),
            'max_length': max(text_lengths),
            'total_documents': len(texts)
        }
        
        # Get vocabulary statistics if available
        if self.vocab is not None:
            stats['vocabulary_size'] = len(self.vocab)
            
        # If we have a count matrix, calculate density
        if self.count_vectorizer is not None:
            try:
                matrix = self.count_vectorizer.transform(texts)
                total_nonzero = matrix.nnz
                total_cells = matrix.shape[0] * matrix.shape[1]
                stats['density'] = total_nonzero / total_cells
            except Exception as e:
                logger.warning(f"Couldn't compute matrix density: {e}")
        
        return stats
    
    def save(self, path: str = None) -> None:
        """Save the NLP processor state to disk.
        
        This method serializes the NLP processor's state, including vectorizers
        and vocabulary, to a specified directory so it can be loaded later.
        
        Args:
            path (str, optional): Directory path to save the processor.
                If None, uses a default path in the model directory.
                
        Returns:
            None
                
        Example:
            >>> processor = NLPProcessor()
            >>> processor.create_document_term_matrix(texts)
            >>> processor.save('models/nlp_processor')
        """
        if path is None:
            path = os.path.join(MODEL_DIR, 'nlp_processor')
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save state dict
        state = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'max_df': self.max_df,
            'random_state': self.random_state,
            'vocab': self.vocab
        }
        
        # Save to files
        joblib.dump(state, f"{path}_state.joblib")
        
        if self.count_vectorizer is not None:
            joblib.dump(self.count_vectorizer, f"{path}_count_vectorizer.joblib")
            
        if self.tfidf_vectorizer is not None:
            joblib.dump(self.tfidf_vectorizer, f"{path}_tfidf_vectorizer.joblib")
            
        logger.info(f"NLP processor saved to {path}")
    
    @classmethod
    def load(cls, path: str = None) -> 'NLPProcessor':
        """Load an NLP processor from disk.
        
        This class method reconstructs an NLPProcessor instance from files
        saved using the save() method. It loads the processor state and any
        saved vectorizers.
        
        Args:
            path (str, optional): Directory path from which to load the processor.
                If None, uses a default path in the model directory.
                
        Returns:
            NLPProcessor: A loaded NLP processor instance
            
        Example:
            >>> processor = NLPProcessor.load('models/nlp_processor')
            >>> dtm, vocab = processor.create_document_term_matrix(new_texts, fit=False)
        """
        if path is None:
            path = os.path.join(MODEL_DIR, 'nlp_processor')
            
        # Load state
        state = joblib.load(f"{path}_state.joblib")
        
        # Create instance
        instance = cls(
            max_features=state['max_features'],
            ngram_range=state['ngram_range'],
            max_df=state['max_df'],
            random_state=state['random_state']
        )
        
        # Load vocabulary
        instance.vocab = state['vocab']
        
        # Load vectorizers if available
        try:
            instance.count_vectorizer = joblib.load(f"{path}_count_vectorizer.joblib")
            logger.info("Loaded count vectorizer")
        except FileNotFoundError:
            logger.info("No count vectorizer found")
            
        try:
            instance.tfidf_vectorizer = joblib.load(f"{path}_tfidf_vectorizer.joblib")
            logger.info("Loaded TF-IDF vectorizer")
        except FileNotFoundError:
            logger.info("No TF-IDF vectorizer found")
            
        logger.info(f"NLP processor loaded from {path}")
        return instance