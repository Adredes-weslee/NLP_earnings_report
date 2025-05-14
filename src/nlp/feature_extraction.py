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
    """Unified feature extractor for financial text analysis.
    
    This class combines multiple NLP techniques to extract structured features 
    from unstructured financial texts. It integrates topic modeling, sentiment analysis,
    named entity recognition (NER), and financial metric extraction into a unified
    feature extraction pipeline.
    
    Attributes:
        use_topics (bool): Whether to include topic-based features.
        use_sentiment (bool): Whether to include sentiment-based features.
        use_metrics (bool): Whether to include extracted financial metrics.
        use_embeddings (bool): Whether to include text embeddings.
        use_spacy (bool): Whether to use spaCy for named entity recognition.
        use_transformers (bool): Whether to use transformer models.
        model_name (str): Name of the model to use (spaCy or transformer).
        topic_model: Topic modeling component (set using set_topic_model).
        sentiment_analyzer: Sentiment analysis component (set using set_sentiment_analyzer).
        embedding_model: Embedding model component (set using set_embedding_model).
        feature_names (list): Names of extracted features.
        feature_importance (dict): Feature importance scores.
    """
    
    def __init__(self, 
                 use_topics: bool = True, 
                 use_sentiment: bool = True, 
                 use_metrics: bool = True, 
                 use_embeddings: bool = False,
                 use_spacy: bool = True, 
                 use_transformers: bool = False,
                 model_name: str = "en_core_web_sm"):
        """Initialize the feature extractor with specified components.
        
        Args:
            use_topics (bool): Whether to include topic features. Defaults to True.
            use_sentiment (bool): Whether to include sentiment features. Defaults to True.
            use_metrics (bool): Whether to include extracted financial metrics. 
                Defaults to True.
            use_embeddings (bool): Whether to include text embeddings. Defaults to False.
            use_spacy (bool): Whether to use spaCy for NER. Defaults to True.
            use_transformers (bool): Whether to use transformers for advanced NER. 
                Defaults to False.
            model_name (str): Name of spaCy model or transformer model to use. 
                Defaults to "en_core_web_sm".
                
        Example:
            >>> extractor = FeatureExtractor(use_embeddings=True)
            >>> extractor.set_topic_model(topic_model)
            >>> features, feature_names = extractor.extract_features(df, 'text_column')
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
    
    def save_with_fallback(self, base_path=None):
        """Save the feature extractor with permission-error handling and fallbacks.
        
        This method tries to save to the standard path first, but if permission issues
        occur, it automatically falls back to a timestamped version and records the
        latest path for future loading.
        
        Args:
            base_path (str, optional): Base directory path for saving. 
                If None, uses the path from config.
            
        Returns:
            str: Path where the model was successfully saved, or None if all saves failed
        """
        import os
        import time
        
        # Use default path if none provided
        if base_path is None:
            base_path = os.path.dirname(FEATURE_EXTRACTOR_PATH)
        
        # Try to create the directory
        os.makedirs(base_path, exist_ok=True)
        
        # First try the standard path
        standard_path = os.path.join(base_path, 'feature_extractor')
        try:
            self.save(standard_path)
            logger.info(f"Feature extractor saved to: {standard_path}")
            
            # Save reference to the latest path
            reference_file = os.path.join(base_path, 'latest_feature_extractor.txt')
            try:
                with open(reference_file, 'w') as f:
                    f.write(standard_path)
            except Exception as e:
                logger.warning(f"Could not save reference file: {e}")
                
            return standard_path
            
        except PermissionError:
            # Fall back to timestamped version
            timestamp = int(time.time())
            alt_path = os.path.join(base_path, f'feature_extractor_{timestamp}')
            
            try:
                self.save(alt_path)
                logger.info(f"Feature extractor saved to timestamped path: {alt_path}")
                
                # Save reference to the latest path
                reference_file = os.path.join(base_path, 'latest_feature_extractor.txt')
                try:
                    with open(reference_file, 'w') as f:
                        f.write(alt_path)
                except Exception as e:
                    logger.warning(f"Could not save reference file: {e}")
                    
                return alt_path
                
            except Exception as e:
                logger.error(f"Failed to save feature extractor: {str(e)}")
                return None

    @classmethod
    def load_latest(cls, base_path=None):
        """Load the latest version of a feature extractor.
        
        This method intelligently finds the most recent feature extractor, looking for:
        1. A reference file pointing to the latest version
        2. The standard non-timestamped path
        3. The most recent timestamped version
        
        Args:
            base_path (str, optional): Base directory to search for models.
                If None, uses the path from config.
            
        Returns:
            FeatureExtractor: Loaded feature extractor or None if not found
        """
        import os
        
        # Use default path if none provided
        if base_path is None:
            base_path = os.path.dirname(FEATURE_EXTRACTOR_PATH)
        
        # Check for reference file first
        reference_file = os.path.join(base_path, 'latest_feature_extractor.txt')
        if os.path.exists(reference_file):
            try:
                with open(reference_file, 'r') as f:
                    model_path = f.read().strip()
                    if os.path.exists(model_path):
                        logger.info(f"Loading feature extractor from reference: {model_path}")
                        return cls.load(model_path)
            except Exception as e:
                logger.warning(f"Could not load from reference file: {e}")
        
        # No reference or reference failed, search for models
        try:
            # Check for standard path
            standard_path = os.path.join(base_path, 'feature_extractor')
            if os.path.exists(standard_path):
                logger.info(f"Loading feature extractor from standard path: {standard_path}")
                return cls.load(standard_path)
            
            # Look for timestamped versions
            if os.path.exists(base_path):
                files = [f for f in os.listdir(base_path) 
                        if f.startswith('feature_extractor_') and os.path.isfile(os.path.join(base_path, f))]
                
                if files:
                    # Sort by timestamp (highest/most recent last)
                    latest_file = sorted(files)[-1]
                    model_path = os.path.join(base_path, latest_file)
                    logger.info(f"Loading feature extractor from latest timestamped path: {model_path}")
                    return cls.load(model_path)
        
        except Exception as e:
            logger.error(f"Error while finding/loading feature extractor: {str(e)}")
        
        logger.error("No feature extractor model found")
        return None
    
    
    def load_spacy_model(self):
        """Load the spaCy NLP model for named entity recognition.
        
        This method loads the specified spaCy model for use in named entity
        recognition tasks if use_spacy is set to True. If successful, 
        it will set the nlp attribute to the loaded model.
        
        Returns:
            None
            
        Raises:
            ImportError: If spaCy is not installed.
            OSError: If the specified model is not found.
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
    
    def extract_financial_metrics(self, text_df: Union[pd.DataFrame, str]) -> Dict[str, float]:
        """Extract financial metrics and ratios from text using regex patterns."""
        # Handle both DataFrame input and direct string input
        if isinstance(text_df, pd.DataFrame):
            if 'text' not in text_df.columns or len(text_df) == 0:
                return {}
            text = text_df['text'].iloc[0]
        else:
            text = text_df
            
        if not isinstance(text, str):
            return {}
        
        logger.info(f"Extracting financial metrics from text: {text[:100]}...")
        
        # Extract basic numerical values even if we can't categorize them
        # This ensures we always return something for demo purposes
        numeric_values = {}
        
        # Find all dollar amounts with more lenient patterns
        dollar_pattern = r'\$\s*(\d+(?:\.\d+)?)'
        dollar_amounts = re.findall(dollar_pattern, text, re.IGNORECASE)
        for i, amount in enumerate(dollar_amounts):
            numeric_values[f'dollar_amount_{i+1}'] = float(amount)
        
        # Find all percentage values
        percent_pattern = r'(\d+(?:\.\d+)?)\s*(?:percent|%)'
        percentages = re.findall(percent_pattern, text, re.IGNORECASE)
        for i, pct in enumerate(percentages):
            numeric_values[f'percentage_{i+1}'] = float(pct)
        
        # If we found any values at all, return them
        if numeric_values:
            logger.info(f"Extracted {len(numeric_values)} numerical values: {list(numeric_values.keys())}")
            return numeric_values
            
        # If we get here, we couldn't extract any metrics
        logger.warning("No financial metrics could be extracted from the text")
        
        # Return a placeholder value to avoid breaking the UI
        return {"no_metrics_found": 0.0}
    
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
        """Extract all enabled features from text data in a DataFrame.
        
        This is the main method for extracting features from text data. It combines
        all enabled feature extraction techniques (topics, sentiment, metrics, 
        embeddings) into a single feature matrix with labeled features.
        
        Args:
            df (pd.DataFrame): DataFrame containing the text data to analyze.
                Should have at least one column with text content.
            text_column (str, optional): Name of the column containing the text
                to analyze. Defaults to 'text'.
            
        Returns:
            Tuple[np.ndarray, List[str]]: A tuple containing:
                - Feature matrix as numpy array with shape (n_samples, n_features)
                - List of feature names corresponding to each column in the matrix
                
        Raises:
            ValueError: If no features are extracted due to missing models or
                if all feature types are disabled
            KeyError: If text_column doesn't exist in the DataFrame
            
        Example:
            >>> # Create a feature extractor with all features enabled
            >>> extractor = FeatureExtractor(
            ...     use_topics=True,
            ...     use_sentiment=True,
            ...     use_metrics=True
            ... )
            >>> # Set required models
            >>> extractor.set_topic_model(topic_model)
            >>> extractor.set_sentiment_analyzer(sentiment_analyzer)
            >>> # Extract features from a DataFrame
            >>> features, feature_names = extractor.extract_features(
            ...     df,
            ...     text_column='earnings_text'
            ... )
            >>> print(f"Extracted {features.shape[1]} features")
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
        """Retrieve the top n most important features by absolute importance.
        
        This method returns the most predictive features based on their importance
        scores (as set by set_feature_importances()). Features are sorted by their
        absolute importance values, allowing both positive and negative features
        to be identified as important.
        
        Args:
            n (int, optional): Number of top features to return. Defaults to 20.
            
        Returns:
            pd.DataFrame: DataFrame with columns:
                - 'feature': Name of the feature
                - 'importance': Importance score of the feature
                Sorted by absolute importance (descending order)
                
        Note:
            This method requires feature importances to be previously set using
            the set_feature_importances() method. Otherwise, it will return an
            empty DataFrame.
            
        Example:
            >>> # Assume feature importances already set from model training
            >>> top_features = extractor.get_top_features(n=10)
            >>> print(top_features)
               feature    importance
            0  topic_5      0.453
            1  sentiment_positive  0.329
            2  sentiment_negative -0.287
            ...
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
        """Organize extracted features into logical groups by type.
        
        This method categorizes the extracted features into semantic groups
        based on their prefixes and content. This is useful for understanding
        which features belong to which analysis types (sentiment, topics, etc.)
        and for feature selection or visualization purposes.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping feature group names to lists
                of feature names. Possible groups include:
                - 'sentiment': Sentiment analysis features
                - 'topics': Topic modeling features
                - 'financial': Extracted financial metrics
                - 'readability': Text readability metrics
                - 'entities': Named entity features
                - 'embeddings': Text embedding features
                Empty groups are excluded from the result.
                
        Example:
            >>> feature_groups = extractor.get_feature_groups()
            >>> for group_name, features in feature_groups.items():
            ...     print(f"{group_name}: {len(features)} features")
            sentiment: 6 features
            topics: 30 features
            financial: 12 features
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
        """Create a horizontal bar chart of feature importances.
        
        This method visualizes the top n most important features as a horizontal
        bar chart, with positive importances in blue and negative importances in red.
        Feature importances must have been previously set using set_feature_importances().
        
        Args:
            n (int, optional): Number of top features to show in the plot.
                Defaults to 20.
            figsize (tuple, optional): Figure size as (width, height) in inches.
                Defaults to (12, 10).
            
        Returns:
            matplotlib.figure.Figure: The generated figure object that can be
                displayed or saved.
                
        Example:
            >>> # Assume feature importances already set
            >>> fig = extractor.plot_feature_importance(n=15)
            >>> fig.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
            >>> plt.show()
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get the importance scores of features used in financial prediction.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        if not hasattr(self, 'feature_importance'):
            # If feature importance wasn't cached during training, create it
            if hasattr(self, 'model') and hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                importances = self.model.feature_importances_
                self.feature_importance = dict(zip(self.feature_names, importances))
            elif hasattr(self, 'model') and hasattr(self.model, 'coef_'):
                # For linear models
                importances = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
                self.feature_importance = dict(zip(self.feature_names, importances))
            else:
                # No feature importance available
                return {}
        
        return self.feature_importance
    
    
    def save(self, path=None):
        """Save the feature extractor configuration and state to disk.
        
        This method serializes the feature extractor's configuration settings,
        feature names, and importance scores to a file on disk. It does not save
        the associated models (topic model, sentiment analyzer, etc.) - these
        must be saved separately.
        
        Args:
            path (str, optional): Path where the feature extractor will be saved.
                If None, uses the default path from config (FEATURE_EXTRACTOR_PATH).
                Defaults to None.
                
        Raises:
            OSError: If there's an error creating the directory or writing the file
            pickle.PicklingError: If the feature extractor cannot be serialized
            
        Example:
            >>> extractor = FeatureExtractor(use_topics=True, use_sentiment=True)
            >>> # Set up models and extract features
            >>> # ...
            >>> extractor.save('models/features/financial_features')
            
        Note:
            To fully restore a feature extractor, you'll need to reload any
            associated models (topic model, sentiment analyzer, etc.) and set
            them using the appropriate methods after loading.
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
        """Load a previously saved feature extractor from disk.
        
        This class method reconstructs a FeatureExtractor instance from a saved
        configuration file. It loads the feature extractor's settings, feature names,
        and importance scores, but NOT the associated models (topic model,
        sentiment analyzer, etc.) - these must be loaded and set separately.
        
        Args:
            path (str, optional): Path to the file containing the saved feature extractor.
                If None, uses the default path from config (FEATURE_EXTRACTOR_PATH).
                Defaults to None.
            
        Returns:
            FeatureExtractor: A new FeatureExtractor instance with the saved
                configuration and state.
                
        Raises:
            FileNotFoundError: If the saved file cannot be found
            pickle.UnpicklingError: If there's an error deserializing the file
            
        Example:
            >>> # Load a previously saved feature extractor
            >>> extractor = FeatureExtractor.load('models/features/financial_features')
            >>> 
            >>> # You'll need to reload and set associated models
            >>> extractor.set_topic_model(topic_model)
            >>> extractor.set_sentiment_analyzer(sentiment_analyzer)
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
