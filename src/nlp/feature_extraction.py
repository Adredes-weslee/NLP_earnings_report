"""
Consolidated feature extraction module for financial text analysis.
Extracts structured features from unstructured earnings reports.
"""

import numpy as np
import pandas as pd
import os
import re
import logging
import pickle
import joblib
import sys
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
import seaborn as sns

# Import configuration values
from ..config import (MAX_FEATURES, NGRAM_RANGE, MAX_DOC_FREQ, NUM_TOPICS,
                  TOPIC_WORD_PRIOR, DOC_TOPIC_PRIOR_FACTOR, RANDOM_STATE,
                  MODEL_DIR, FEATURE_EXTRACTOR_PATH)

# Optional imports for advanced feature extraction
try:
    import spacy
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger('feature_extractor')

class FeatureExtractor:
    """
    Unified feature extractor for financial text analysis.
    Combines topic modeling, sentiment analysis, NER, and financial metric extraction.
    """
    
    def __init__(self, 
                 use_topics: bool = True, 
                 use_sentiment: bool = True, 
                 use_metrics: bool = True, 
                 use_embeddings: bool = False,
                 use_spacy: bool = True, 
                 use_transformers: bool = False,
                 model_name: str = "en_core_web_sm"):
        """
        Initialize the feature extractor.
        
        Args:
            use_topics: Whether to include topic features
            use_sentiment: Whether to include sentiment features
            use_metrics: Whether to include extracted financial metrics
            use_embeddings: Whether to include text embeddings
            use_spacy: Whether to use spaCy for NER
            use_transformers: Whether to use transformers for advanced NER
            model_name: Name of spaCy model or transformer model to use
        """
        self.use_topics = use_topics
        self.use_sentiment = use_sentiment
        self.use_metrics = use_metrics
        self.use_embeddings = use_embeddings
        self.use_spacy = use_spacy
        self.use_transformers = use_transformers
        self.model_name = model_name
        
        # Components that can be set later
        self.topic_model = None
        self.sentiment_analyzer = None
        self.embedding_model = None
        self.vectorizer = None
        
        # NLP-specific attributes
        self.nlp = None
        self.ner_pipeline = None
        
        # Feature extraction results
        self.feature_names = None
        self.feature_importance = None
        
        logger.info(f"Initialized FeatureExtractor with: topics={use_topics}, "
                  f"sentiment={use_sentiment}, metrics={use_metrics}, embeddings={use_embeddings}")
        
    def load_spacy_model(self):
        """
        Load the spaCy NLP model for named entity recognition.
        """
        if not self.use_spacy:
            return
            
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.use_spacy = False
    
    def load_transformer_ner(self):
        """
        Load transformer-based NER pipeline.
        """
        if not self.use_transformers or not TRANSFORMERS_AVAILABLE:
            return
            
        try:
            self.ner_pipeline = pipeline('ner', model=self.model_name)
            logger.info(f"Loaded transformer NER model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load transformer NER model: {e}")
            self.use_transformers = False
    
    def extract_financial_metrics(self, text: str) -> Dict[str, float]:
        """
        Extract financial metrics and ratios from text using regex patterns.
        
        Args:
            text (str): Input financial text
            
        Returns:
            dict: Dictionary of extracted financial metrics
        """
        if not isinstance(text, str):
            return {}
        
        metrics = {}
        
        # Patterns for common financial metrics with units
        patterns = {
            'revenue_million': r'revenue (?:of )?\$?(\d+(?:\.\d+)?)\s*million',
            'revenue_billion': r'revenue (?:of )?\$?(\d+(?:\.\d+)?)\s*billion',
            'gross_margin': r'gross margin (?:of )?(\d+(?:\.\d+)?)%',
            'operating_margin': r'operating margin (?:of )?(\d+(?:\.\d+)?)%',
            'profit_margin': r'(?:profit|net) margin (?:of )?(\d+(?:\.\d+)?)%',
            'eps': r'(?:EPS|earnings per share) (?:of )?\$?(\d+(?:\.\d+)?)',
            'diluted_eps': r'diluted (?:EPS|earnings per share) (?:of )?\$?(\d+(?:\.\d+)?)',
            'yoy_growth': r'(?:year[- ]over[- ]year|y-o-y|yoy) growth (?:of )?(\d+(?:\.\d+)?)%',
            'qoq_growth': r'(?:quarter[- ]over[- ]quarter|q-o-q|qoq) growth (?:of )?(\d+(?:\.\d+)?)%',
            'cash': r'cash (?:and cash equivalents )?(?:of )?\$?(\d+(?:\.\d+)?)\s*(?:million|billion)',
            'debt': r'(?:debt|loans) (?:of )?\$?(\d+(?:\.\d+)?)\s*(?:million|billion)',
        }
        
        # Extract metrics
        for name, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                # If multiple matches, use suffix to distinguish
                key = f"{name}_{i+1}" if i > 0 else name
                value = float(match.group(1))
                metrics[key] = value
        
        return metrics
    
    def extract_metrics_from_texts(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract financial metrics from a list of texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            tuple: (feature matrix, feature names)
        """
        if not self.use_metrics:
            return None, []
            
        metric_features = []
        all_keys = set()
        all_metrics = []
        
        # First pass: extract metrics and collect all keys
        for text in texts:
            metrics = self.extract_financial_metrics(text)
            all_keys.update(metrics.keys())
            all_metrics.append(metrics)
        
        feature_names = sorted(list(all_keys))
        
        # Second pass: create feature matrix with consistent columns
        for metrics in all_metrics:
            values = [metrics.get(name, 0.0) for name in feature_names]
            metric_features.append(values)
        
        return np.array(metric_features), feature_names
    def create_document_term_matrix(self, texts: List[str], save_path: str = None) -> Tuple:
        """
        Create a document-term matrix from cleaned texts
        
        Args:
            texts: Cleaned text data
            save_path: Path to save the vectorizer
            
        Returns:
            tuple: (document-term matrix, vectorizer object, feature names)
        """
        from nltk.corpus import stopwords
        stops = stopwords.words('english')
        
        vec = CountVectorizer(
            token_pattern=r'\b[a-zA-Z_]{3,}[a-zA-Z]*\b',
            ngram_range=NGRAM_RANGE,
            max_features=MAX_FEATURES,            stop_words=stops,
            max_df=MAX_DOC_FREQ
        )
        
        dtm = vec.fit_transform(texts)
        vocab = vec.get_feature_names_out()
        
        logger.info(f"DTM shape (documents x features): {dtm.shape}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(vec, f)
            logger.info(f"Vectorizer saved to {save_path}")
        
        self.vectorizer = vec
        return dtm, vec, vocab
    
    def set_topic_model(self, model):
        """Set the topic model to use for feature extraction"""
        self.topic_model = model
        return self
    
    def set_sentiment_analyzer(self, analyzer):
        """Set the sentiment analyzer to use for feature extraction"""
        self.sentiment_analyzer = analyzer
        return self
    
    def set_embedding_model(self, model, vectorizer=None):
        """Set the embedding model to use for text embeddings"""
        self.embedding_model = model
        if vectorizer:
            self.vectorizer = vectorizer
        return self
    
    def extract_topic_features(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract topic distribution features from texts"""
        if not self.use_topics or not self.topic_model:
            logger.warning("Topic model not set or disabled, skipping topic feature extraction")
            return None, []
        
        # Log model expected feature count for debugging
        expected_model_features = 0
        if hasattr(self.topic_model.model, 'components_'):
            expected_model_features = self.topic_model.model.components_.shape[1]
            logger.info(f"Topic model expects features: {expected_model_features}")
        
        # CRITICAL FIX: Use the vectorizer from the topic model if available
        if hasattr(self.topic_model, 'vectorizer') and self.topic_model.vectorizer is not None:
            # Use the vectorizer that was used to train the topic model
            dtm = self.topic_model.vectorizer.transform(texts)
            logger.info(f"Using topic model vectorizer, DTM shape: {dtm.shape}")
        elif hasattr(self.embedding_model, 'vectorizer') and self.embedding_model.vectorizer is not None:
            # Fall back to embedding vectorizer if available
            dtm = self.embedding_model.vectorizer.transform(texts)
            logger.info(f"Using embedding model vectorizer, DTM shape: {dtm.shape}")
        else:
            # Only create a new vectorizer as a last resort
            # And make sure to use the same parameters as when training the topic model
            logger.warning("No existing vectorizer found, creating a new one. This may cause dimension mismatches.")
            
            # Try to match the dimensions of the trained model
            expected_features = expected_model_features if expected_model_features > 0 else MAX_FEATURES
            logger.info(f"Creating new DTM with expected features: {expected_features}")
            
            dtm, _, _ = self.create_document_term_matrix(texts, max_features=expected_features)
        
        try:
            topic_distributions = self.topic_model.transform(dtm)
            logger.info(f"Topic distributions shape: {topic_distributions.shape}")
        except Exception as e:
            logger.error(f"Error transforming DTM: {e}. DTM shape: {dtm.shape}")
            if expected_model_features > 0:
                logger.error(f"Topic model expects {expected_model_features} features, but received {dtm.shape[1]}")
            raise
        
        # Create feature names for topics
        topic_feature_names = [f"topic_{i}" for i in range(topic_distributions.shape[1])]
        
        return topic_distributions, topic_feature_names
    
    def extract_sentiment_features(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract sentiment features from texts"""
        if not self.use_sentiment or not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not set or disabled, skipping sentiment feature extraction")
            return None, []
        
        sentiment_features = []
        feature_names = []
        
        for text in texts:
            sentiment = self.sentiment_analyzer.analyze(text)
            sentiment_features.append(list(sentiment.values()))
            
            # Set feature names on first iteration
            if not feature_names:
                feature_names = [f"sentiment_{k}" for k in sentiment.keys()]
        
        return np.array(sentiment_features), feature_names
    
    def extract_embedding_features(self, texts: List[str], reduce_dim: int = None) -> Tuple[np.ndarray, List[str]]:
        """Extract text embedding features"""
        if not self.use_embeddings or not self.embedding_model:
            logger.warning("Embedding model not set or disabled, skipping embedding feature extraction")
            return None, []
        
        # Generate embeddings
        embeddings = self.embedding_model.transform(texts)
        
        # Reduce dimensionality if requested
        if reduce_dim and reduce_dim < embeddings.shape[1]:
            svd = TruncatedSVD(n_components=reduce_dim, random_state=42)
            embeddings = svd.fit_transform(embeddings)
            feature_names = [f"embedding_{i}" for i in range(reduce_dim)]
        else:
            feature_names = [f"embedding_{i}" for i in range(embeddings.shape[1])]
        
        return embeddings, feature_names
    
    def extract_features(self, df, text_column='text') -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features from text data.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of column containing text
            
        Returns:
            tuple: (feature matrix, feature names)
        """
        texts = df[text_column].fillna('').tolist()
        features = []
        all_feature_names = []
        
        # Extract topic features if enabled
        if self.use_topics and self.topic_model:
            topic_features, topic_names = self.extract_topic_features(texts)
            if topic_features is not None:
                features.append(topic_features)
                all_feature_names.extend(topic_names)
                logger.info(f"Added {len(topic_names)} topic features")
        
        # Extract sentiment features if enabled
        if self.use_sentiment and self.sentiment_analyzer:
            sentiment_features, sentiment_names = self.extract_sentiment_features(texts)
            if sentiment_features is not None:
                features.append(sentiment_features)
                all_feature_names.extend(sentiment_names)
                logger.info(f"Added {len(sentiment_names)} sentiment features")
        
        # Extract financial metric features if enabled
        if self.use_metrics:
            metric_features, metric_names = self.extract_metrics_from_texts(texts)
            if metric_features is not None:
                features.append(metric_features)
                all_feature_names.extend(metric_names)
                logger.info(f"Added {len(metric_names)} financial metric features")
        
        # Extract embedding features if enabled
        if self.use_embeddings and self.embedding_model:
            embedding_features, embedding_names = self.extract_embedding_features(texts, reduce_dim=50)
            if embedding_features is not None:
                features.append(embedding_features)
                all_feature_names.extend(embedding_names)
                logger.info(f"Added {len(embedding_names)} embedding features")
        
        if not features:
            raise ValueError("No features extracted. Check feature extraction settings and models.")
        
        # Combine all feature sets
        combined_features = np.hstack(features)
        self.feature_names = all_feature_names
        
        logger.info(f"Total features extracted: {combined_features.shape[1]}")
        return combined_features, all_feature_names
    
    def combine_features(self, topic_features=None, sentiment_features=None, 
                        financial_features=None, embeddings=None) -> Tuple[np.ndarray, List[str]]:
        """
        Combine different types of features into a single feature matrix.
        
        Args:
            topic_features: Topic distribution features
            sentiment_features: Sentiment analysis features
            financial_features: Extracted financial metrics
            embeddings: Text embedding features
            
        Returns:
            tuple: (Combined feature matrix, List of feature names)
        """
        features_to_combine = []
        feature_names = []
        
        # Add topic features if provided
        if topic_features is not None:
            features_to_combine.append(topic_features)
            feature_names.extend([f"topic_{i}" for i in range(topic_features.shape[1])])
            logger.info(f"Added {topic_features.shape[1]} topic features")
        
        # Add sentiment features if provided
        if sentiment_features is not None:
            features_to_combine.append(sentiment_features)
            if isinstance(sentiment_features, pd.DataFrame):
                feature_names.extend(sentiment_features.columns)
                sentiment_features = sentiment_features.values
            else:
                feature_names.extend([f"sentiment_{i}" for i in range(sentiment_features.shape[1])])
            logger.info(f"Added {sentiment_features.shape[1]} sentiment features")
        
        # Add financial features if provided
        if financial_features is not None:
            features_to_combine.append(financial_features)
            if isinstance(financial_features, pd.DataFrame):
                feature_names.extend(financial_features.columns)
                financial_features = financial_features.values
            else:
                feature_names.extend([f"financial_{i}" for i in range(financial_features.shape[1])])
            logger.info(f"Added {financial_features.shape[1]} financial features")
        
        # Add embedding features if provided
        if embeddings is not None:
            # Optionally reduce dimensionality if embeddings are too large
            if embeddings.shape[1] > 50:
                svd = TruncatedSVD(n_components=50, random_state=42)
                embeddings = svd.fit_transform(embeddings)
                logger.info(f"Reduced embedding dimensions from {embeddings.shape[1]} to 50")
            
            features_to_combine.append(embeddings)
            feature_names.extend([f"embedding_{i}" for i in range(embeddings.shape[1])])
            logger.info(f"Added {embeddings.shape[1]} embedding features")
        
        if not features_to_combine:
            raise ValueError("No features provided to combine")
        
        # Handle case of only one feature set
        if len(features_to_combine) == 1:
            return features_to_combine[0], feature_names
        
        # Otherwise combine all features horizontally
        combined_features = np.hstack(features_to_combine)
        logger.info(f"Combined feature matrix shape: {combined_features.shape}")
        
        return combined_features, feature_names
    
    def set_feature_importances(self, importances, feature_names=None):
        """
        Set feature importance values.
        
        Args:
            importances: Array of feature importance values
            feature_names: Feature names (optional)
        """
        names = feature_names if feature_names is not None else self.feature_names
        
        if names is None or len(importances) != len(names):
            logger.warning("Warning: Feature names don't match importance values")
            self.feature_importance = {i: importance for i, importance in enumerate(importances)}
        else:
            self.feature_importance = {name: importance for name, importance in zip(names, importances)}
        
        return self
    
    def get_top_features(self, n=20):
        """
        Get the top n most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            pandas.DataFrame: DataFrame with feature names and importance values
        """
        if not self.feature_importance:
            logger.warning("Feature importances not set. Use set_feature_importances first.")
            return pd.DataFrame(columns=["feature", "importance"])
        
        # Create sorted DataFrame of feature importances
        importance_df = pd.DataFrame({
            "feature": list(self.feature_importance.keys()),
            "importance": list(self.feature_importance.values())
        })
        
        # Sort by absolute importance since negative values can be important too
        importance_df["abs_importance"] = importance_df["importance"].abs()
        importance_df = importance_df.sort_values("abs_importance", ascending=False).head(n)
        importance_df = importance_df.drop("abs_importance", axis=1)
        
        return importance_df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get groups of features organized by type.
        
        Returns:
            Dictionary mapping feature group names to lists of feature names
        """
        feature_groups = {
            'sentiment': [],
            'readability': [],
            'financial': [],
            'entities': [],
            'topics': [],
            'embeddings': []
        }
        
        # Add available features to their respective groups
        for feature in self.feature_names if self.feature_names else []:
            if feature.startswith('sentiment_'):
                feature_groups['sentiment'].append(feature)
            elif feature.startswith('topic_'):
                feature_groups['topics'].append(feature)
            elif feature.startswith('entity_'):
                feature_groups['entities'].append(feature)
            elif feature.startswith('embedding_'):
                feature_groups['embeddings'].append(feature)
            elif any(f in feature for f in ['revenue', 'earnings', 'eps', 'margin', 'growth', 'cash', 'debt']):
                feature_groups['financial'].append(feature)
            elif any(f in feature for f in ['reading', 'sentence', 'syllable', 'fog']):
                feature_groups['readability'].append(feature)
        
        # Remove empty groups
        return {k: v for k, v in feature_groups.items() if v}
    
    def plot_feature_importance(self, n=20, figsize=(12, 10)):
        """
        Plot feature importances.
        
        Args:
            n: Number of top features to show
            figsize: Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        top_features = self.get_top_features(n)
        
        plt.figure(figsize=figsize)
        colors = ['red' if x < 0 else 'blue' for x in top_features['importance']]
        plt.barh(y=top_features['feature'], width=top_features['importance'], color=colors)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {n} Feature Importances')
        plt.tight_layout()
        
        return plt.gcf()
        
    def save(self, path=None):
        """
        Save the feature extractor.
        
        Args:
            path: Path to save the feature extractor. If None, use default from config.
        """
        if path is None:
            path = FEATURE_EXTRACTOR_PATH
            
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create a dictionary with the state to save
            state = {
                'use_topics': self.use_topics,
                'use_sentiment': self.use_sentiment,
                'use_metrics': self.use_metrics,
                'use_embeddings': self.use_embeddings,
                'use_spacy': self.use_spacy,
                'use_transformers': self.use_transformers,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance
            }
            
            # Save the state
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Feature extractor saved to {path}")
        except Exception as e:
            logger.error(f"Error saving feature extractor: {str(e)}")
            # Re-raise the exception so the caller can handle it
            raise
        
    @classmethod
    def load(cls, path=None):
        """
        Load a feature extractor.
        
        Args:
            path: Path to load the feature extractor from. If None, use default from config.
            
        Returns:
            FeatureExtractor: Loaded feature extractor
        """
        if path is None:
            path = FEATURE_EXTRACTOR_PATH
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance with the saved settings
        instance = cls(
            use_topics=state.get('use_topics', True),
            use_sentiment=state.get('use_sentiment', True),
            use_metrics=state.get('use_metrics', True),
            use_embeddings=state.get('use_embeddings', False),
            use_spacy=state.get('use_spacy', True),
            use_transformers=state.get('use_transformers', False)
        )
        
        # Restore state
        instance.feature_names = state.get('feature_names')
        instance.feature_importance = state.get('feature_importance')
        
        return instance
