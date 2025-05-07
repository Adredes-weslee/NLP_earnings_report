"""
Feature extraction module for NLP earnings report analysis.
Combines different text representations and features.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union, Optional, Tuple
import joblib
import os
from scipy import sparse

logger = logging.getLogger('feature_extractor')

class FeatureExtractor:
    """
    Enhanced feature extractor that combines multiple types of features:
    - Embeddings (TF-IDF, transformers)
    - Topic distributions
    - Sentiment scores
    - Financial metrics
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.embedding_processor = None
        self.sentiment_analyzer = None
        self.topic_modeler = None
        self.feature_names = []
        self.feature_groups = {}
        self.feature_importances = {}
        
    def set_embedding_processor(self, embedding_processor):
        """Set the embedding processor component."""
        self.embedding_processor = embedding_processor
        return self
    
    def set_sentiment_analyzer(self, sentiment_analyzer):
        """Set the sentiment analyzer component."""
        self.sentiment_analyzer = sentiment_analyzer
        return self
    
    def set_topic_modeler(self, topic_modeler):
        """Set the topic modeler component."""
        self.topic_modeler = topic_modeler
        return self
    
    def extract_features(self, df: pd.DataFrame, text_column: str = 'processed_text', 
                         include_embeddings: bool = True, 
                         include_topics: bool = True,
                         include_sentiment: bool = True,
                         include_financial_metrics: bool = False,
                         financial_columns: List[str] = None) -> Tuple[sparse.csr_matrix, List[str]]:
        """
        Extract multiple types of features and combine them.
        
        Args:
            df: DataFrame containing texts and other features
            text_column: Name of column containing processed text
            include_embeddings: Whether to include embeddings
            include_topics: Whether to include topic distributions
            include_sentiment: Whether to include sentiment scores
            include_financial_metrics: Whether to include financial metrics
            financial_columns: List of financial metric columns to include
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        feature_matrices = []
        self.feature_names = []
        self.feature_groups = {}
        start_idx = 0
        
        # 1. Extract embeddings
        if include_embeddings and self.embedding_processor is not None:
            logger.info("Extracting text embeddings")
            try:
                X_embeddings = self.embedding_processor.get_document_embeddings(df, text_column)
                
                # If sparse, convert to csr_matrix for consistency
                if sparse.issparse(X_embeddings):
                    X_embeddings = X_embeddings.tocsr()
                else:
                    X_embeddings = sparse.csr_matrix(X_embeddings)
                    
                feature_matrices.append(X_embeddings)
                
                # Create feature names for embeddings
                if hasattr(self.embedding_processor, 'vocab') and self.embedding_processor.vocab is not None:
                    emb_feature_names = list(self.embedding_processor.vocab)
                else:
                    emb_feature_names = [f"emb_{i}" for i in range(X_embeddings.shape[1])]
                
                self.feature_names.extend(emb_feature_names)
                end_idx = start_idx + X_embeddings.shape[1]
                self.feature_groups['embeddings'] = (start_idx, end_idx)
                start_idx = end_idx
                
                logger.info(f"Added {X_embeddings.shape[1]} embedding features")
            except Exception as e:
                logger.error(f"Error extracting embeddings: {str(e)}")
        
        # 2. Extract topic distributions
        if include_topics and self.topic_modeler is not None:
            logger.info("Extracting topic distributions")
            try:
                if self.topic_modeler.method == 'gensim_lda':
                    # For gensim models, we need to convert tokens to document-term matrix
                    # This should be handled before calling extract_features
                    pass
                
                # If we already have DTM from embeddings, use that
                dtm = None
                if hasattr(self.embedding_processor, 'vectorizer') and self.embedding_processor.vectorizer is not None:
                    # If we're using BoW or TF-IDF embeddings, we can reuse the DTM
                    if self.embedding_processor.method in ['bow', 'tfidf']:
                        tokens = df[text_column].fillna('').tolist()
                        dtm = self.embedding_processor.vectorizer.transform(tokens)
                
                # Get topic distributions
                if dtm is not None and hasattr(self.topic_modeler, 'model'):
                    X_topics = self.topic_modeler.get_document_topics(dtm)
                else:
                    # If we don't have a DTM, try to get topic distributions directly
                    # This assumes the topic modeler has been fitted
                    X_topics = self.topic_modeler.get_document_topics()
                
                # Ensure it's CSR format
                X_topics = sparse.csr_matrix(X_topics)
                feature_matrices.append(X_topics)
                
                # Create feature names for topics
                topic_feature_names = [f"topic_{i}" for i in range(X_topics.shape[1])]
                self.feature_names.extend(topic_feature_names)
                
                end_idx = start_idx + X_topics.shape[1]
                self.feature_groups['topics'] = (start_idx, end_idx)
                start_idx = end_idx
                
                logger.info(f"Added {X_topics.shape[1]} topic features")
            except Exception as e:
                logger.error(f"Error extracting topics: {str(e)}")
                
        # 3. Extract sentiment scores
        if include_sentiment and self.sentiment_analyzer is not None:
            logger.info("Extracting sentiment features")
            try:
                # Get sentiment scores
                sentiment_results = self.sentiment_analyzer.batch_analyze(df[text_column].fillna('').tolist())
                
                # Convert to sparse matrix
                X_sentiment = sparse.csr_matrix(sentiment_results.values)
                feature_matrices.append(X_sentiment)
                
                # Create feature names for sentiment
                sentiment_feature_names = list(sentiment_results.columns)
                self.feature_names.extend(sentiment_feature_names)
                
                end_idx = start_idx + X_sentiment.shape[1]
                self.feature_groups['sentiment'] = (start_idx, end_idx)
                start_idx = end_idx
                
                logger.info(f"Added {X_sentiment.shape[1]} sentiment features")
            except Exception as e:
                logger.error(f"Error extracting sentiment: {str(e)}")
        
        # 4. Extract financial metrics
        if include_financial_metrics and financial_columns is not None:
            logger.info("Extracting financial metrics")
            try:
                # Get financial metric columns
                financial_data = df[financial_columns].fillna(0)
                
                # Convert to sparse matrix
                X_financial = sparse.csr_matrix(financial_data.values)
                feature_matrices.append(X_financial)
                
                # Create feature names for financial metrics
                self.feature_names.extend(financial_columns)
                
                end_idx = start_idx + X_financial.shape[1]
                self.feature_groups['financial'] = (start_idx, end_idx)
                start_idx = end_idx
                
                logger.info(f"Added {X_financial.shape[1]} financial metric features")
            except Exception as e:
                logger.error(f"Error extracting financial metrics: {str(e)}")
        
        # Combine all features
        if not feature_matrices:
            raise ValueError("No features were extracted")
            
        if len(feature_matrices) == 1:
            X = feature_matrices[0]
        else:
            X = sparse.hstack(feature_matrices, format='csr')
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        
        return X, self.feature_names
    
    def get_feature_groups(self) -> Dict[str, Tuple[int, int]]:
        """
        Get feature group index ranges.
        
        Returns:
            Dictionary mapping feature group names to (start_idx, end_idx) tuples
        """
        return self.feature_groups
    
    def set_feature_importances(self, importances: np.ndarray, feature_names=None):
        """
        Set feature importances from a trained model.
        
        Args:
            importances: Array of feature importance scores
            feature_names: Optional list of feature names (if different from self.feature_names)
        """
        names = feature_names if feature_names is not None else self.feature_names
        if len(names) != len(importances):
            logger.warning(f"Feature importances length ({len(importances)}) doesn't match feature names ({len(names)})")
            return
            
        self.feature_importances = dict(zip(names, importances))
        
        # Also calculate aggregate importance by feature group
        group_importances = {}
        for group_name, (start_idx, end_idx) in self.feature_groups.items():
            group_importance = np.sum(importances[start_idx:end_idx])
            group_importances[group_name] = group_importance
            
        self.group_importances = group_importances
        
        return self
    
    def get_top_features(self, n=20) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.feature_importances:
            logger.warning("Feature importances not set")
            return pd.DataFrame()
            
        # Convert to DataFrame and sort
        importances_df = pd.DataFrame({
            'feature': list(self.feature_importances.keys()),
            'importance': list(self.feature_importances.values())
        })
        
        # Sort by absolute importance (direction doesn't matter for feature ranking)
        importances_df['abs_importance'] = importances_df['importance'].abs()
        importances_df = importances_df.sort_values('abs_importance', ascending=False)
        
        # Get top N features
        top_features = importances_df.head(n).drop('abs_importance', axis=1)
        
        return top_features
    
    def get_group_importances(self) -> pd.DataFrame:
        """
        Get importance scores aggregated by feature group.
        
        Returns:
            DataFrame with group names and importance scores
        """
        if not hasattr(self, 'group_importances') or not self.group_importances:
            logger.warning("Group importances not calculated")
            return pd.DataFrame()
            
        # Convert to DataFrame and sort
        group_imp_df = pd.DataFrame({
            'feature_group': list(self.group_importances.keys()),
            'importance': list(self.group_importances.values())
        })
        
        # Sort by absolute importance
        group_imp_df['abs_importance'] = group_imp_df['importance'].abs()
        group_imp_df = group_imp_df.sort_values('abs_importance', ascending=False)
        
        return group_imp_df.drop('abs_importance', axis=1)
    
    def plot_feature_importances(self, n=20, figsize=(10, 8)):
        """
        Plot top N feature importances.
        
        Args:
            n: Number of top features to display
            figsize: Figure size as (width, height) tuple
        """
        top_features = self.get_top_features(n=n)
        
        if top_features.empty:
            logger.warning("No feature importances to plot")
            return None
            
        plt.figure(figsize=figsize)
        
        # Sort for the plot (ascending order for horizontal bar plot)
        top_features = top_features.sort_values('importance')
        
        # Create horizontal bar plot
        bars = plt.barh(top_features['feature'], top_features['importance'])
        
        # Color bars by sign
        for i, bar in enumerate(bars):
            if top_features['importance'].iloc[i] < 0:
                bar.set_color('r')
            else:
                bar.set_color('g')
                
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {n} Feature Importances')
        plt.grid(axis='x')
        plt.tight_layout()
        
        return plt.gcf()
    
    def save(self, path: str) -> None:
        """
        Save feature extractor configuration.
        
        Args:
            path: Directory path to save configuration
        """
        os.makedirs(path, exist_ok=True)
        
        # Save feature names and groups
        data = {
            'feature_names': self.feature_names,
            'feature_groups': self.feature_groups,
            'feature_importances': self.feature_importances
        }
        
        if hasattr(self, 'group_importances'):
            data['group_importances'] = self.group_importances
            
        joblib.dump(data, os.path.join(path, 'feature_extractor_data.joblib'))
        logger.info(f"Feature extractor data saved to {path}")
        
        # Note: We don't save the component objects themselves,
        # as they should be saved separately
    
    @classmethod
    def load(cls, path: str) -> 'FeatureExtractor':
        """
        Load feature extractor configuration.
        
        Args:
            path: Directory path to load configuration from
            
        Returns:
            Loaded FeatureExtractor instance
        """
        data = joblib.load(os.path.join(path, 'feature_extractor_data.joblib'))
        
        instance = cls()
        instance.feature_names = data['feature_names']
        instance.feature_groups = data['feature_groups']
        instance.feature_importances = data['feature_importances']
        
        if 'group_importances' in data:
            instance.group_importances = data['group_importances']
            
        logger.info(f"Feature extractor loaded from {path}")
        return instance