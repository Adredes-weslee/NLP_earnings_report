"""
Feature extraction module for earnings report text analysis.
Extracts structured features from unstructured financial text.
"""

import numpy as np
import pandas as pd
import os
import re
import logging
import joblib
from typing import List, Dict, Union, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# Optional imports for advanced feature extraction
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger('feature_extractor')

class FeatureExtractor:
    """
    Class for extracting structured features from financial text.
    Extracts numerical metrics, entities, and other key features from earnings reports.
    """
    
    def __init__(self, use_spacy: bool = True, use_transformers: bool = False,
                model_name: str = "en_core_web_sm"):
        """
        Initialize the feature extractor.
        
        Args:
            use_spacy: Whether to use spaCy for NER
            use_transformers: Whether to use transformers for advanced NER
            model_name: Name of spaCy model or transformer model to use
        """
        self.use_spacy = use_spacy
        self.use_transformers = use_transformers
        self.model_name = model_name
        self.nlp = None
        self.ner_pipeline = None
        self.feature_importance = {}
        
        logger.info(f"Initializing FeatureExtractor with use_spacy={use_spacy}, "
                   f"use_transformers={use_transformers}")
        
        # Initialize spaCy if requested
        if use_spacy:
            self._load_spacy()
        
        # Initialize transformers if requested
        if use_transformers:
            self._load_transformers()
    
    def _load_spacy(self):
        """Load spaCy NER model."""
        try:
            import spacy
            if not spacy.util.is_package(self.model_name):
                logger.warning(f"spaCy model '{self.model_name}' not found, downloading...")
                spacy.cli.download(self.model_name)
            
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            self.use_spacy = False
    
    def _load_transformers(self):
        """Load transformers NER pipeline."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. Cannot use transformers for NER.")
            self.use_transformers = False
            return
        
        try:
            self.ner_pipeline = pipeline(
                "token-classification", 
                model=self.model_name,
                aggregation_strategy="simple"
            )
            logger.info(f"Loaded transformers NER model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading transformers model: {str(e)}")
            self.use_transformers = False
    
    def extract_numerical_metrics(self, text: str) -> Dict[str, float]:
        """
        Extract numerical metrics from financial text.
        
        Args:
            text: Input financial text
            
        Returns:
            Dictionary of extracted metrics and their values
        """
        if not isinstance(text, str):
            return {}
        
        # Dictionary to store extracted metrics
        metrics = {}
        
        # Regular expressions for common financial metrics
        patterns = {
            # Revenue patterns
            'revenue': r'(?:revenue|revenues|sales) of \$?(\d+(?:\.\d+)?)\s*(?:million|billion|m|b|k)?',
            # Earnings/profit patterns
            'earnings': r'(?:earnings|profit|income|ebitda) of \$?(\d+(?:\.\d+)?)\s*(?:million|billion|m|b|k)?',
            # EPS patterns
            'eps': r'(?:earnings per share|eps) of \$?(\d+(?:\.\d+)?)',
            # Growth patterns
            'growth': r'(?:growth|increase|grew|up) (?:by |of )?(\d+(?:\.\d+)?)%',
            # Margin patterns
            'margin': r'(?:margin|margins) of (\d+(?:\.\d+)?)%',
            # Market share
            'market_share': r'(?:market share|share) of (\d+(?:\.\d+)?)%',
            # Guidance
            'guidance': r'(?:guidance|forecast|expect|expects) .*?\$?(\d+(?:\.\d+)?)\s*(?:million|billion|m|b|k)?'
        }
        
        # Extract metrics using regex patterns
        for metric, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                value_str = match.group(1)
                try:
                    value = float(value_str)
                    if i == 0:
                        metrics[metric] = value
                    else:
                        # If multiple matches, create indexed metrics
                        metrics[f"{metric}_{i+1}"] = value
                except ValueError:
                    continue
        
        return metrics
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy or transformers.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if not isinstance(text, str):
            return {}
        
        entities = {}
        
        if self.use_spacy and self.nlp:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
        
        elif self.use_transformers and self.ner_pipeline:
            # Process text with transformers
            result = self.ner_pipeline(text)
            
            # Extract entities
            for item in result:
                entity_type = item['entity_group']
                entity_text = item['word']
                
                if entity_type not in entities:
                    entities[entity_type] = []
                if entity_text not in entities[entity_type]:
                    entities.append(entity_text)
        
        return entities
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment-related features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment features
        """
        if not isinstance(text, str):
            return {}
        
        features = {}
        
        # Simple lexicon-based approach
        positive_words = ['increase', 'growth', 'profit', 'gain', 'improved', 'positive',
                          'strong', 'success', 'exceeded', 'better', 'record']
        negative_words = ['decrease', 'decline', 'loss', 'dropped', 'negative', 'weak',
                          'challenging', 'failed', 'below', 'worse', 'disappointing']
        
        # Count word occurrences
        text_lower = text.lower()
        positive_count = sum(text_lower.count(' ' + word + ' ') for word in positive_words)
        negative_count = sum(text_lower.count(' ' + word + ' ') for word in negative_words)
        
        # Calculate simple sentiment metrics
        total_count = positive_count + negative_count
        if total_count > 0:
            features['sentiment_polarity'] = (positive_count - negative_count) / total_count
            features['sentiment_ratio'] = positive_count / (negative_count + 1)  # Avoid div by zero
        else:
            features['sentiment_polarity'] = 0
            features['sentiment_ratio'] = 1
        
        features['positive_word_count'] = positive_count
        features['negative_word_count'] = negative_count
        
        return features
    
    def extract_readability_features(self, text: str) -> Dict[str, float]:
        """
        Extract readability metrics from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of readability features
        """
        if not isinstance(text, str) or not text.strip():
            return {}
        
        # Split into sentences and words
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text)
        
        # Calculate basic metrics
        num_sentences = len(sentences)
        num_words = len(words)
        avg_sentence_len = num_words / num_sentences if num_sentences > 0 else 0
        
        # Calculate syllables (approximation)
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            for char in word:
                if char in vowels:
                    if not prev_is_vowel:
                        count += 1
                    prev_is_vowel = True
                else:
                    prev_is_vowel = False
            # Adjust for silent e at end
            if word.endswith('e') and count > 1:
                count -= 1
            return max(1, count)
        
        syllable_counts = [count_syllables(word) for word in words]
        num_syllables = sum(syllable_counts)
        avg_syllables_per_word = num_syllables / num_words if num_words > 0 else 0
        
        # Calculate Flesch Reading Ease
        # Higher scores are easier to read (90-100: Very easy, 0-30: Very difficult)
        flesch = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables_per_word)
        
        # Calculate Gunning Fog Index
        # Estimates years of formal education needed to understand the text
        complex_words = sum(1 for count in syllable_counts if count >= 3)
        complex_pct = complex_words / num_words if num_words > 0 else 0
        fog = 0.4 * (avg_sentence_len + 100 * complex_pct)
        
        # Return readability metrics
        return {
            'avg_sentence_length': avg_sentence_len,
            'avg_syllables_per_word': avg_syllables_per_word,
            'flesch_reading_ease': flesch,
            'gunning_fog_index': fog,
            'complex_word_pct': complex_pct * 100,  # Convert to percentage
        }
    
    def extract_financial_metrics(self, text: str) -> Dict[str, float]:
        """
        Extract financial metrics and ratios from text.
        
        Args:
            text: Input financial text
            
        Returns:
            Dictionary of financial metrics
        """
        metrics = {}
        
        # Patterns for common financial metrics with units
        patterns = {
            # Revenue with units
            'revenue_million': r'revenue (?:of )?\$?(\d+(?:\.\d+)?)\s*million',
            'revenue_billion': r'revenue (?:of )?\$?(\d+(?:\.\d+)?)\s*billion',
            # Profit margins
            'gross_margin': r'gross margin (?:of )?(\d+(?:\.\d+)?)%',
            'operating_margin': r'operating margin (?:of )?(\d+(?:\.\d+)?)%',
            'profit_margin': r'(?:profit|net) margin (?:of )?(\d+(?:\.\d+)?)%',
            # EPS related
            'eps': r'(?:EPS|earnings per share) (?:of )?\$?(\d+(?:\.\d+)?)',
            'diluted_eps': r'diluted (?:EPS|earnings per share) (?:of )?\$?(\d+(?:\.\d+)?)',
            # Growth rates
            'yoy_growth': r'(?:year[- ]over[- ]year|y-o-y|yoy) growth (?:of )?(\d+(?:\.\d+)?)%',
            'qoq_growth': r'(?:quarter[- ]over[- ]quarter|q-o-q|qoq) growth (?:of )?(\d+(?:\.\d+)?)%',
            # Cash and debt
            'cash': r'cash (?:and cash equivalents )?(?:of )?\$?(\d+(?:\.\d+)?)\s*(?:million|billion)',
            'debt': r'(?:debt|loans) (?:of )?\$?(\d+(?:\.\d+)?)\s*(?:million|billion)',
        }
        
        # Extract metrics
        for name, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1)
                try:
                    value = float(value_str)
                    # Convert billions to millions for consistency
                    if 'billion' in name:
                        value = value * 1000
                    metrics[name] = value
                except ValueError:
                    continue
        
        return metrics
    
    def extract_features(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Extract all features from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of all extracted features
        """
        if not isinstance(text, str):
            return {}
        
        # Combine all feature extraction methods
        features = {}
        
        # Extract numerical metrics
        numerical_metrics = self.extract_numerical_metrics(text)
        features.update(numerical_metrics)
        
        # Extract sentiment features
        sentiment_features = self.extract_sentiment_features(text)
        features.update(sentiment_features)
        
        # Extract readability features
        readability_features = self.extract_readability_features(text)
        features.update(readability_features)
        
        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(text)
        features.update(financial_metrics)
        
        # Extract named entities (add top entities only to avoid too many features)
        entities = self.extract_named_entities(text)
        for entity_type, entity_list in entities.items():
            if entity_list:
                # Add only first entity of each type
                features[f"entity_{entity_type}"] = entity_list[0]
        
        return features
    
    def fit(self, texts: List[str], targets=None) -> 'FeatureExtractor':
        """
        Fit the feature extractor (extract feature importance if targets provided).
        
        Args:
            texts: List of text documents
            targets: Optional target values for feature importance
            
        Returns:
            Self for method chaining
        """
        if not texts:
            logger.warning("Empty text list provided to fit()")
            return self
        
        logger.info(f"Fitting feature extractor on {len(texts)} texts")
        
        # Extract features for all texts
        all_features = []
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
            
            features = self.extract_features(text)
            all_features.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Calculate feature importance if targets provided
        if targets is not None and len(targets) == len(texts):
            # Use correlation with target for feature importance
            for col in features_df.columns:
                if pd.api.types.is_numeric_dtype(features_df[col]):
                    corr = features_df[col].corr(targets)
                    self.feature_importance[col] = abs(corr)  # Use absolute correlation
        
        return self
    
    def transform(self, texts: List[str]) -> pd.DataFrame:
        """
        Transform texts to feature matrix.
        
        Args:
            texts: List of text documents
            
        Returns:
            DataFrame of extracted features
        """
        if not texts:
            logger.warning("Empty text list provided to transform()")
            return pd.DataFrame()
        
        logger.info(f"Transforming {len(texts)} texts to features")
        
        # Extract features for all texts
        all_features = []
        for text in texts:
            features = self.extract_features(text)
            all_features.append(features)
        
        # Convert to DataFrame
        return pd.DataFrame(all_features)
    
    def fit_transform(self, texts: List[str], targets=None) -> pd.DataFrame:
        """
        Fit and transform texts in one step.
        
        Args:
            texts: List of text documents
            targets: Optional target values for feature importance
            
        Returns:
            DataFrame of extracted features
        """
        self.fit(texts, targets)
        return self.transform(texts)
    
    def plot_feature_importance(self, top_n: int = 10) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if not self.feature_importance:
            logger.warning("No feature importance available. Call fit() with targets first.")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No feature importance available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': list(self.feature_importance.keys()),
            'Importance': list(self.feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.set_xlabel('Importance (Absolute Correlation)')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        
        return fig
    
    def set_embedding_processor(self, embedding_processor):
        """
        Set the embedding processor to use for embedding-based features.
        
        Args:
            embedding_processor: EmbeddingProcessor instance to use
        """
        self.embedding_processor = embedding_processor
        logger.info(f"Embedding processor set to {type(embedding_processor).__name__}")
        return self

    def save(self, path: str) -> None:
        """
        Save the feature extractor.
        
        Args:
            path: Directory path to save model
        """
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            'use_spacy': self.use_spacy,
            'use_transformers': self.use_transformers,
            'model_name': self.model_name,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(config, os.path.join(path, 'feature_extractor_config.joblib'))
        logger.info(f"Feature extractor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureExtractor':
        """
        Load a feature extractor from disk.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded FeatureExtractor instance
        """
        config = joblib.load(os.path.join(path, 'feature_extractor_config.joblib'))
        
        instance = cls(
            use_spacy=config['use_spacy'],
            use_transformers=config['use_transformers'],
            model_name=config['model_name']
        )
        
        instance.feature_importance = config['feature_importance']
        
        logger.info(f"Feature extractor loaded from {path}")
        return instance