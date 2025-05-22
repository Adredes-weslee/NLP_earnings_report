"""
Consolidated feature extraction module for financial text analysis.
Extracts structured features from unstructured earnings reports.
"""
import numpy as np
import pandas as pd
import logging
import os
import joblib
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as LDA

# Import configuration values
from ..config import (MAX_FEATURES, NGRAM_RANGE, MAX_DOC_FREQ, NUM_TOPICS,
                  TOPIC_WORD_PRIOR, DOC_TOPIC_PRIOR_FACTOR, RANDOM_STATE,
                  MODEL_DIR, FEATURE_EXTRACTOR_PATH)

# Import the centralized NLPProcessor
from .nlp_processing import NLPProcessor

# Optional imports for advanced feature extraction
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger('feature_extractor')

class FeatureExtractor:
    """Extract structured features from financial text for analysis and modeling.
    
    This class extracts various text features that can be used for downstream 
    machine learning tasks like classification or regression. It uses the 
    centralized NLPProcessor for vectorization to ensure consistent processing
    across the pipeline.
    
    Features extracted include:
    - Basic text statistics (length, word counts, etc.)
    - Topic-based features using LDA
    - Semantic features using SVD (LSA)
    - Advanced features using transformers when available
    
    Attributes:
        max_features (int): Maximum vocabulary size for vectorization
        ngram_range (tuple): Range of n-grams to include in features
        random_state (int): Random seed for reproducibility
        nlp_processor (NLPProcessor): Centralized processor for vectorization
        svd_model: Trained SVD model for semantic feature extraction
        lda_model: Trained LDA model for topic feature extraction
    """
    
    def __init__(self, max_features: int = MAX_FEATURES,
                 ngram_range: Tuple[int, int] = NGRAM_RANGE,
                 max_df: float = MAX_DOC_FREQ,
                 random_state: int = RANDOM_STATE,
                 nlp_processor: NLPProcessor = None):
        """Initialize the feature extractor with specified parameters.
        
        Args:
            max_features (int, optional): Maximum vocabulary size for vectorization.
                Defaults to MAX_FEATURES from config.
            ngram_range (tuple, optional): Range of n-grams to include in features.
                Defaults to NGRAM_RANGE from config.
            max_df (float, optional): Maximum document frequency for terms.
                Defaults to MAX_DOC_FREQ from config.
            random_state (int, optional): Random seed for reproducibility.
                Defaults to RANDOM_STATE from config.
            nlp_processor (NLPProcessor, optional): Centralized NLP processor for
                vectorization. If None, a new one will be created when needed.
                
        Example:
            >>> # Using shared NLP processor
            >>> nlp_proc = NLPProcessor(max_features=5000)
            >>> extractor = FeatureExtractor(nlp_processor=nlp_proc)
            >>> features = extractor.extract_features(texts)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.random_state = random_state
        self.nlp_processor = nlp_processor
        
        # Models for feature extraction
        self.svd_model = None
        self.lda_model = None
        
        logger.info(f"Initialized FeatureExtractor with max_features={max_features}")
    
    def extract_financial_metrics(self, data):
        """Extract financial metrics from text.
        
        This method identifies and extracts financial metrics like revenue,
        profit, EPS, etc. from earnings report text.
        
        Args:
            text (str): Earnings report text to analyze
            
        Returns:
            dict: Dictionary of extracted financial metrics
            
        Example:
            >>> metrics = extractor.extract_financial_metrics(text)
            >>> print(f"Revenue: {metrics.get('revenue')}")
        """
        import re
        
        # Process input - handle both DataFrame and string inputs
        if isinstance(data, pd.DataFrame):
            if 'text' in data.columns and not data.empty:
                text = data['text'].iloc[0]
            else:
                text = str(data.iloc[0]) if not data.empty else ""
        else:
            text = str(data)
        
        # Basic patterns for financial metrics
        patterns = {
            'revenue_dollar': r'\$?\s*(\d+(?:\.\d+)?)\s*(?:million|billion|m|b)?\s*(?:in)?\s*(?:revenue|sales)',
            'profit_dollar': r'\$?\s*(\d+(?:\.\d+)?)\s*(?:million|billion|m|b)?\s*(?:in)?\s*(?:profit|net income)',
            'eps_dollar': r'(?:earnings per share|eps).{1,50}?\$?\s*(\d+\.\d+)',
            'growth_percentage': r'(?:growth|increase).{1,20}?\s*(\d+(?:\.\d+)?)\s*%',
            'margin_percentage': r'(?:margin).{1,20}?\s*(\d+(?:\.\d+)?)\s*%',
        }
        
        results = {}
        
        # Apply regex patterns to extract metrics
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                # Take the first match for simplicity
                results[metric] = matches[0]
        
        logger.info(f"Extracted {len(results)} financial metrics")
        return results
    
    def create_document_term_matrix(self, texts: List[str], vectorizer_type: str = 'count') -> Tuple:
        """Create a document-term matrix from the provided texts.
        
        This method delegates to the centralized NLPProcessor to create
        document-term matrices with consistent vectorization settings.
        
        Args:
            texts (List[str]): List of text documents to vectorize
            vectorizer_type (str, optional): Type of vectorizer to use:
                'count' for term frequency, 'tfidf' for TF-IDF weighting.
                Defaults to 'count'.
                
        Returns:
            Tuple: A tuple containing:
                - document-term matrix (sparse matrix)
                - list of vocabulary terms (feature names)
                
        Example:
            >>> dtm, vocab = extractor.create_document_term_matrix(texts, vectorizer_type='tfidf')
            >>> print(f"Matrix shape: {dtm.shape}, Vocabulary size: {len(vocab)}")
                
        Note:
            This method uses the centralized NLPProcessor for vectorization when available,
            ensuring consistent tokenization across the pipeline.
        """
        # Create NLPProcessor if not available
        if self.nlp_processor is None:
            self.nlp_processor = NLPProcessor(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                max_df=self.max_df,
                random_state=self.random_state
            )
            logger.info("Created new NLPProcessor for document-term matrix creation")
        
        # Delegate to the centralized NLPProcessor
        if vectorizer_type == 'tfidf':
            dtm, vocab = self.nlp_processor.create_tfidf_matrix(texts)
        else:
            dtm, vocab = self.nlp_processor.create_document_term_matrix(texts)
            
        logger.info(f"Created {vectorizer_type} matrix with shape {dtm.shape}")
        
        return dtm, vocab
    
    def extract_semantic_features(self, dtm, n_components: int = 100, 
                                 algorithm: str = 'randomized') -> np.ndarray:
        """Extract semantic features using SVD (Latent Semantic Analysis).
        
        This method performs dimensionality reduction on the document-term matrix
        to identify latent semantic features in the text.
        
        Args:
            dtm: Document-term matrix (sparse or dense)
            n_components (int, optional): Number of components (dimensions) to extract.
                Defaults to 100.
            algorithm (str, optional): SVD algorithm to use ('randomized' or 'arpack').
                Defaults to 'randomized'.
                
        Returns:
            np.ndarray: Matrix of semantic features with shape (n_documents, n_components)
                
        Example:
            >>> dtm, _ = extractor.create_document_term_matrix(texts)
            >>> semantic_features = extractor.extract_semantic_features(dtm, n_components=50)
            >>> print(f"Semantic feature shape: {semantic_features.shape}")
        """
        logger.info(f"Extracting {n_components} semantic features using SVD")
        
        # Create and fit the SVD model
        self.svd_model = TruncatedSVD(
            n_components=n_components,
            algorithm=algorithm,
            random_state=self.random_state
        )
        
        # Transform the DTM to semantic features
        semantic_features = self.svd_model.fit_transform(dtm)
        
        variance_explained = self.svd_model.explained_variance_ratio_.sum()
        logger.info(f"Extracted semantic features explain {variance_explained:.2%} of variance")
        
        return semantic_features
    
    def extract_topic_features(self, dtm, n_topics: int = NUM_TOPICS) -> np.ndarray:
        """Extract topic-based features using LDA.
        
        This method identifies topics in the document collection and returns
        the topic distributions for each document as features.
        
        Args:
            dtm: Document-term matrix (sparse or dense)
            n_topics (int, optional): Number of topics to extract.
                Defaults to NUM_TOPICS from config.
                
        Returns:
            np.ndarray: Topic distribution matrix with shape (n_documents, n_topics)
                
        Example:
            >>> dtm, _ = extractor.create_document_term_matrix(texts)
            >>> topic_features = extractor.extract_topic_features(dtm, n_topics=20)
            >>> print(f"Document-topic distribution shape: {topic_features.shape}")
        """
        logger.info(f"Extracting {n_topics} topic features using LDA")
        
        # Calculate document-topic prior parameter
        doc_topic_prior = min(1.0, DOC_TOPIC_PRIOR_FACTOR / n_topics)
        
        # Create and fit the LDA model
        self.lda_model = LDA(
            n_components=n_topics,
            topic_word_prior=TOPIC_WORD_PRIOR,
            doc_topic_prior=doc_topic_prior,
            max_iter=20,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Transform the DTM to topic distributions
        topic_distributions = self.lda_model.fit_transform(dtm)
        
        logger.info(f"Extracted {n_topics} topic distributions")
        return topic_distributions
    
    def extract_statistical_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract statistical features from texts.
        
        This method calculates various statistical metrics about the texts,
        such as word count, sentence count, and readability scores.
        
        Args:
            texts (List[str]): List of text documents
                
        Returns:
            pd.DataFrame: DataFrame containing statistical features with one row
                per document and columns for each feature
                
        Example:
            >>> stats_df = extractor.extract_statistical_features(texts)
            >>> print(stats_df.head())
        """
        features = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
                
            # Basic length features
            words = text.split()
            sentences = text.split('.')
            
            # Calculate features
            feat = {
                'word_count': len(words),
                'char_count': len(text),
                'sentence_count': max(1, len(sentences)),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'avg_sentence_length': len(words) / max(1, len(sentences)),
            }
            
            features.append(feat)
            
        return pd.DataFrame(features)
    
    def extract_transformer_features(self, texts: List[str], 
                                   model_name: str = 'distilbert-base-uncased') -> np.ndarray:
        """Extract features using transformer models.
        
        This method uses pre-trained transformer models to generate rich
        text embeddings that capture semantic meaning.
        
        Args:
            texts (List[str]): List of text documents
            model_name (str, optional): Name of the transformer model to use.
                Defaults to 'distilbert-base-uncased'.
                
        Returns:
            np.ndarray: Feature matrix with transformer embeddings
                
        Raises:
            ImportError: If transformers library is not available
            
        Example:
            >>> if TRANSFORMERS_AVAILABLE:
            ...     transformer_features = extractor.extract_transformer_features(texts)
            ...     print(f"Transformer feature shape: {transformer_features.shape}")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Install with: pip install transformers")
            
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        logger.info(f"Extracting transformer features using {model_name}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Process in batches to avoid memory issues
        batch_size = 8
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and get model inputs
            encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt",
                               max_length=512)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**encoded)
                
            # Use the CLS token as the document embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(batch_embeddings)
            
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated transformer embeddings with shape {embeddings.shape}")
        return embeddings
    
    def extract_features(self, texts: List[str], include_statistical: bool = True,
                       include_semantic: bool = True, include_topics: bool = True,
                       include_transformer: bool = False,
                       semantic_components: int = 50,
                       n_topics: int = NUM_TOPICS) -> Dict[str, np.ndarray]:
        """Extract multiple feature sets from text documents.
        
        This method extracts multiple types of features and returns them
        in a dictionary. It serves as a central entry point for comprehensive
        feature extraction.
        
        Args:
            texts (List[str]): List of text documents
            include_statistical (bool, optional): Whether to include statistical features.
                Defaults to True.
            include_semantic (bool, optional): Whether to include semantic features (SVD).
                Defaults to True.
            include_topics (bool, optional): Whether to include topic features (LDA).
                Defaults to True.
            include_transformer (bool, optional): Whether to include transformer embeddings.
                Requires transformers library. Defaults to False.
            semantic_components (int, optional): Number of semantic components to extract.
                Defaults to 50.
            n_topics (int, optional): Number of topics to extract.
                Defaults to NUM_TOPICS from config.
                
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping feature set names to feature matrices
                
        Example:
            >>> features = extractor.extract_features(
            ...     texts, 
            ...     include_transformer=True,
            ...     semantic_components=100
            ... )
            >>> for name, matrix in features.items():
            ...     print(f"{name} shape: {matrix.shape}")
        """
        features = {}
        
        # Create document-term matrix using NLPProcessor
        dtm, vocab = self.create_document_term_matrix(texts)
        
        if include_statistical:
            stats_df = self.extract_statistical_features(texts)
            features['statistical'] = stats_df.values
            
        if include_semantic:
            semantic = self.extract_semantic_features(dtm, n_components=semantic_components)
            features['semantic'] = semantic
            
        if include_topics:
            topics = self.extract_topic_features(dtm, n_topics=n_topics)
            features['topics'] = topics
            
        if include_transformer and TRANSFORMERS_AVAILABLE:
            try:
                transformer = self.extract_transformer_features(texts)
                features['transformer'] = transformer
            except Exception as e:
                logger.warning(f"Failed to extract transformer features: {e}")
                
        # Always include the DTM for completeness
        features['dtm'] = dtm
        
        return features
    
    def combine_features(self, feature_dict: Dict[str, np.ndarray], 
                       feature_sets: List[str] = None) -> np.ndarray:
        """Combine multiple feature sets into a single feature matrix.
        
        This method combines selected feature sets from the dictionary returned
        by extract_features() into a single feature matrix for modeling.
        
        Args:
            feature_dict (Dict[str, np.ndarray]): Dictionary of feature matrices
            feature_sets (List[str], optional): List of feature set names to combine.
                If None, uses all available feature sets except 'dtm'.
                Defaults to None.
                
        Returns:
            np.ndarray: Combined feature matrix
                
        Example:
            >>> features = extractor.extract_features(texts)
            >>> # Combine semantic and topic features only
            >>> combined = extractor.combine_features(features, ['semantic', 'topics'])
            >>> print(f"Combined feature shape: {combined.shape}")
        """
        if feature_sets is None:
            # Use all feature sets except the raw DTM
            feature_sets = [name for name in feature_dict.keys() if name != 'dtm']
            
        # Check that all requested feature sets exist
        for name in feature_sets:
            if name not in feature_dict:
                raise KeyError(f"Feature set '{name}' not found in feature dictionary")
        
        # Combine features horizontally (column-wise)
        matrices = [feature_dict[name] for name in feature_sets]
        combined = np.hstack(matrices)
        
        logger.info(f"Combined features from {feature_sets} with shape {combined.shape}")
        return combined
    
    def save(self, path: str = None) -> None:
        """Save the feature extractor and models to disk.
        
        This method saves the feature extractor's configuration and trained
        models to the specified path for later use.
        
        Args:
            path (str, optional): Path to save the feature extractor.
                If None, uses FEATURE_EXTRACTOR_PATH from config.
                
        Example:
            >>> extractor.extract_features(texts)  # Train models
            >>> extractor.save('models/feature_extractor')
        """
        if path is None:
            path = FEATURE_EXTRACTOR_PATH
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save configuration
        config = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'max_df': self.max_df,
            'random_state': self.random_state
        }
        
        with open(f"{path}_config.joblib", 'wb') as f:
            joblib.dump(config, f)
            
        # Save models if they exist
        if self.svd_model is not None:
            with open(f"{path}_svd_model.joblib", 'wb') as f:
                joblib.dump(self.svd_model, f)
                
        if self.lda_model is not None:
            with open(f"{path}_lda_model.joblib", 'wb') as f:
                joblib.dump(self.lda_model, f)
                
        # Save the NLPProcessor if available
        if self.nlp_processor is not None:
            self.nlp_processor.save(f"{path}_nlp_processor")
            
        logger.info(f"Feature extractor saved to {path}")
    
    @classmethod
    def load(cls, path: str = None) -> 'FeatureExtractor':
        """Load a feature extractor from disk.
        
        This class method reconstructs a FeatureExtractor instance from
        saved files, including configuration and trained models.
        
        Args:
            path (str, optional): Path to load the feature extractor from.
                If None, uses FEATURE_EXTRACTOR_PATH from config.
                
        Returns:
            FeatureExtractor: Loaded feature extractor instance
                
        Example:
            >>> extractor = FeatureExtractor.load('models/feature_extractor')
            >>> features = extractor.extract_features(new_texts)
        """
        if path is None:
            path = FEATURE_EXTRACTOR_PATH
            
        # Load configuration
        with open(f"{path}_config.joblib", 'rb') as f:
            config = joblib.load(f)
            
        # Try to load the NLPProcessor if available
        nlp_processor = None
        try:
            from .nlp_processing import NLPProcessor
            nlp_processor = NLPProcessor.load(f"{path}_nlp_processor")
            logger.info("Loaded NLPProcessor")
        except FileNotFoundError:
            logger.info("No NLPProcessor found")
        except Exception as e:
            logger.warning(f"Error loading NLPProcessor: {str(e)}")
            
        # Create instance
        instance = cls(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            max_df=config['max_df'],
            random_state=config['random_state'],
            nlp_processor=nlp_processor
        )
        
        # Load models if they exist
        try:
            with open(f"{path}_svd_model.joblib", 'rb') as f:
                instance.svd_model = joblib.load(f)
        except FileNotFoundError:
            logger.info("No SVD model found")
            
        try:
            with open(f"{path}_lda_model.joblib", 'rb') as f:
                instance.lda_model = joblib.load(f)
        except FileNotFoundError:
            logger.info("No LDA model found")
            
        logger.info(f"Feature extractor loaded from {path}")
        return instance