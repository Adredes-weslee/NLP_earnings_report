"""
Utility functions for the NLP earnings report dashboard.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import pickle
import io
import base64

# Add the parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import NLP components
from nlp.embedding import EmbeddingProcessor
from nlp.sentiment import SentimentAnalyzer
from nlp.topic_modeling import TopicModeler
from nlp.feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard_utils.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dashboard_utils')

def load_models() -> Dict[str, Any]:
    """
    Load pre-trained models for use in the dashboard.
    
    Returns:
        Dict[str, Any]: Dictionary containing loaded models and data
    """
    models = {}
    
    try:
        # Try to load sample data
        try:
            sample_data_path = "data/processed/train_edad7fda80.csv"
            if os.path.exists(sample_data_path):
                models['sample_data'] = pd.read_csv(sample_data_path)
                logger.info(f"Loaded sample data: {len(models['sample_data'])} samples")
            else:
                # Try to load from data directory
                data_files = [f for f in os.listdir("data/processed") 
                              if f.endswith('.csv') and f.startswith('train_')]
                if data_files:
                    models['sample_data'] = pd.read_csv(f"data/processed/{data_files[0]}")
                    logger.info(f"Loaded alternative sample data: {len(models['sample_data'])} samples")
        except Exception as e:
            logger.warning(f"Could not load sample data: {str(e)}")
        
        # Load embedding model
        try:
            embedding_dir = "models/embeddings"
            if os.path.exists(embedding_dir):
                embedding_models = [f for f in os.listdir(embedding_dir) 
                                   if os.path.isdir(os.path.join(embedding_dir, f))]
                if embedding_models:
                    models['embedding'] = EmbeddingProcessor.load(os.path.join(embedding_dir, embedding_models[0]))
                    logger.info(f"Loaded embedding model: {embedding_models[0]}")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {str(e)}")
        
        # Load sentiment model
        try:
            sentiment_dir = "models/sentiment"
            if os.path.exists(sentiment_dir):
                sentiment_models = [f for f in os.listdir(sentiment_dir) 
                                   if os.path.isdir(os.path.join(sentiment_dir, f))]
                if sentiment_models:
                    models['sentiment'] = SentimentAnalyzer.load(os.path.join(sentiment_dir, sentiment_models[0]))
                    logger.info(f"Loaded sentiment model: {sentiment_models[0]}")
                else:
                    # Create a default sentiment analyzer
                    models['sentiment'] = SentimentAnalyzer(method="loughran_mcdonald")
                    logger.info("Created default LM sentiment analyzer")
        except Exception as e:
            logger.warning(f"Could not load sentiment model: {str(e)}")
            # Create a default sentiment analyzer
            models['sentiment'] = SentimentAnalyzer(method="loughran_mcdonald")
        
        # Load topic model
        try:
            topic_dir = "models/topics"
            if os.path.exists(topic_dir):
                topic_models = [f for f in os.listdir(topic_dir) 
                               if os.path.isdir(os.path.join(topic_dir, f))]
                if topic_models:
                    models['topic'] = TopicModeler.load(os.path.join(topic_dir, topic_models[0]))
                    logger.info(f"Loaded topic model: {topic_models[0]}")
        except Exception as e:
            logger.warning(f"Could not load topic model: {str(e)}")
        
        # Load feature extractor
        try:
            feature_dir = "models/features"
            if os.path.exists(feature_dir):
                feature_models = [f for f in os.listdir(feature_dir) 
                                 if os.path.isdir(os.path.join(feature_dir, f))]
                if feature_models:
                    models['feature_extractor'] = FeatureExtractor.load(os.path.join(feature_dir, feature_models[0]))
                    logger.info(f"Loaded feature extractor: {feature_models[0]}")
        except Exception as e:
            logger.warning(f"Could not load feature extractor: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
    
    return models

def format_topics(topic_model: TopicModeler, num_topics: int = 10, num_words: int = 10) -> pd.DataFrame:
    """
    Format topic model output for display.
    
    Args:
        topic_model: Trained topic model
        num_topics: Number of topics to display
        num_words: Number of words per topic to display
    
    Returns:
        pd.DataFrame: DataFrame with formatted topic information
    """
    if topic_model is None:
        return pd.DataFrame()
    
    num_topics = min(num_topics, topic_model.num_topics)
    
    try:
        topics_df = pd.DataFrame(columns=['Topic ID', 'Top Words', 'Coherence'])
        
        for i in range(num_topics):
            top_words = topic_model.get_topic_words(i, num_words)
            if isinstance(top_words, list):
                words_str = ", ".join(top_words)
            else:
                # Handle case where we get (word, score) tuples
                words_str = ", ".join([f"{word}" for word in top_words])
            
            coherence = topic_model.get_topic_coherence(i) if hasattr(topic_model, 'get_topic_coherence') else None
            
            topics_df = pd.concat([topics_df, pd.DataFrame({
                'Topic ID': [f"Topic {i}"],
                'Top Words': [words_str],
                'Coherence': [f"{coherence:.4f}" if coherence is not None else "N/A"]
            })], ignore_index=True)
        
        return topics_df
    
    except Exception as e:
        logger.error(f"Error formatting topics: {str(e)}")
        return pd.DataFrame()

def classify_sentiment(text: str, sentiment_analyzer: SentimentAnalyzer) -> Dict[str, float]:
    """
    Analyze sentiment of a text.
    
    Args:
        text: Text to analyze
        sentiment_analyzer: Sentiment analyzer to use
    
    Returns:
        Dict[str, float]: Dictionary of sentiment scores
    """
    if not text or sentiment_analyzer is None:
        return {}
    
    try:
        result = sentiment_analyzer.analyze(text)
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {}

def format_sentiment_result(sentiment_result: Dict[str, float]) -> pd.DataFrame:
    """
    Format sentiment analysis results for display.
    
    Args:
        sentiment_result: Results from sentiment analysis
    
    Returns:
        pd.DataFrame: Formatted sentiment results
    """
    if not sentiment_result:
        return pd.DataFrame({'Dimension': ['No results'], 'Score': [0.0]})
    
    try:
        # Format the results as a DataFrame
        result_df = pd.DataFrame({
            'Dimension': list(sentiment_result.keys()),
            'Score': list(sentiment_result.values())
        })
        
        # Sort by score magnitude
        result_df['AbsScore'] = result_df['Score'].abs()
        result_df = result_df.sort_values('AbsScore', ascending=False).drop('AbsScore', axis=1)
        
        return result_df
    except Exception as e:
        logger.error(f"Error formatting sentiment result: {str(e)}")
        return pd.DataFrame({'Error': ['Error formatting results']})

def extract_topic_visualization(topic_model: TopicModeler) -> Optional[str]:
    """
    Extract interactive visualization HTML for topics.
    
    Args:
        topic_model: Topic model
    
    Returns:
        Optional[str]: HTML visualization or None
    """
    if topic_model is None or not hasattr(topic_model, 'get_visualization_html'):
        return None
    
    try:
        html = topic_model.get_visualization_html()
        return html
    except Exception as e:
        logger.error(f"Error getting topic visualization: {str(e)}")
        return None

def get_feature_importance_plot(feature_extractor: FeatureExtractor, n: int = 20) -> Optional[plt.Figure]:
    """
    Generate a plot of feature importances.
    
    Args:
        feature_extractor: Feature extractor with importances
        n: Number of top features to show
    
    Returns:
        Optional[plt.Figure]: Matplotlib figure or None
    """
    if feature_extractor is None or not hasattr(feature_extractor, 'feature_importances'):
        return None
    
    try:
        # Get top features by absolute importance
        top_features = feature_extractor.get_top_features(n=n)
        
        if top_features is None or top_features.empty:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        colors = ['green' if x > 0 else 'red' for x in top_features['importance']]
        bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
        
        # Add labels
        ax.set_title(f'Top {n} Feature Importances', fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 if width > 0 else width - 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', va='center', ha='left' if width > 0 else 'right',
                   fontsize=8)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        return None